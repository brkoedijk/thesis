"""
Deep Hedging Example Worlds
---------------------------
Example world for deep hedging.

June 30, 2022
@author: hansbuehler
"""

import math as math
from collections.abc import Mapping

import numpy as np
from cdxbasics.dynaplot import colors_tableau, figure
from cdxbasics.util import uniqueHash
from scipy.optimize import bisect, minimize
import seaborn as sns
import matplotlib.pyplot as plt

# from tqdm import tqdm
from scipy.stats import norm
from scipy.spatial.distance import cdist

from deephedging.base import (
    DIM_DUMMY,
    Config,
    Logger,
    assert_iter_not_is_nan,
    dh_dtype,
    pdct,
    tf,
    tf_dict,
)
from deephedging.parameters import (
    DEFAULT_A_AGG,
    DEFAULT_A_MATRIX,
    DEFAULT_SIGMA_AGG,
    DEFAULT_SIGMA_MATRIX,
)

_log = Logger(__file__)


class SimpleWorld_Spot_ATM(object):
    """
    Simple World with one asset and one floating ATM option.
    The asset has stochastic volatility, and a mean-reverting drift.
    The implied volatility of the asset is not the realized volatility.

    * To use black & scholes mode use hard overwrite black_scholes = True
    * To turn off stochastic vol use no_stoch_vol = True
    * To turn off mean reverrsion of the drift set no_stoch_drift = True

    Members
    -------
        clone()
            Create a clone of this world with a different seed as validation set

    Attributes
    ----------
        data : dict
            Numpy data of the world

            market : dict
                Dictionary of market data with second dimension equal to step size (numpy)

            features : dict
                per_step : dict - Dictionary of features with second dimension equal to step size (numpy)
                per_path : dict - Dictionary of features valid per path

        tf_data : dict
            Returns a dictionary of TF tensors of 'data' for the use of gym.call() or train()

        tf_y : tf.Tensor
            y data for gym.call() or train(), usually a dummy vector.

        sample_weights : np.ndarray
            sample weights for manual calculations outside tensorflow
            Dimension (nSamples,)

        tf_sample_weights : tf.Tensor:
            sample weights for train()
            Dimension (nSamples,1) c.f. https://stackoverflow.com/questions/60399983/how-to-create-and-use-weighted-metrics-in-keras

        details : dict
            Dictionary of details, e.g. the hidden drift and realized vol of the asset (numpy)
            Most important usually is 'spot_all' which are all spots of the equity (not options), including at maturity (e.g. it has nSteps+1)

        nSamples : int
            Number of samples

        nSteps : int
            Number of steps

        nInst : int
            Number of instruments.

        dt : floast
            Time step.
            TODO: remove in favour of the timeline below

        timelime : np.ndarray
            Generalized timeline. Includes last time point T, e.g. is of length nSteps+1

        config : Config
            Copy of the config file, for cloning

        unique_id : str
            Unique ID generate off the config file, for serialization
    """

    def __init__(self, config: Config, dtype=dh_dtype):
        """
        Parameters
        ----------
        config : Config
            Long list. Use the report feature of 'config' for full feature set

                config  = Config()
                world   = SimpleWorld_Spot_ATM(config)
                print( config.usage_report( with_values = False )

             To use black & scholes mode use hard overwrite black_scholes = True
        """
        self.tf_dtype = dtype
        self.np_dtype = dtype.as_numpy_dtype()
        self.unique_id = None  # for serialization; see below
        self.config = config.copy()  # for cloning

        # simulator
        # ---------
        nSteps = config("steps", 10, int, help="Number of time steps")
        nSamples = config("samples", 1000, int, help="Number of samples")
        seed = config("seed", 2312414312, int, help="Random seed")
        nIvSteps = config(
            "invar_steps",
            5,
            int,
            help="Number of steps ahead to sample from invariant distribution",
        )
        dt = config(
            "dt",
            1.0 / 50.0,
            float,
            help="Time per timestep.",
            help_default="One week (1/50)",
        )
        cost_s = config("cost_s", 0.0002, float, help="Trading cost spot")
        ubnd_as = config(
            "ubnd_as",
            5.0,
            float,
            help="Upper bound for the number of shares traded at each time step",
        )
        lbnd_as = config(
            "lbnd_as",
            -5.0,
            float,
            help="Lower bound for the number of shares traded at each time step",
        )
        bs_mode = config(
            "black_scholes",
            False,
            bool,
            help="Hard overwrite to use a black & scholes model with vol 'rvol' and drift 'drift'. Also turns off the option as a tradable instrument by setting strike = 0.",
        )
        no_svol = config(
            "no_stoch_vol",
            False,
            bool,
            help="If true, turns off stochastic realized and implied vol, by setting meanrev_*vol = 0 and volvol_*vol = 0",
        )
        no_sdrift = config(
            "no_stoch_drift",
            False,
            bool,
            help="If true, turns off the stochastic drift of the asset, by setting meanrev_drift = 0. and drift_vol = 0",
        )
        _log.verify(nSteps > 0, "'steps' must be positive; found %ld", nSteps)
        _log.verify(nSamples > 0, "'samples' must be positive; found %ld", nSamples)
        _log.verify(dt > 0.0, "dt must be positive; found %g", dt)
        _log.verify(cost_s >= 0, "'cost_s' must not be negative; found %g", cost_s)
        _log.verify(ubnd_as >= 0.0, "'ubnd_as' must not be negative; found %g", ubnd_as)
        _log.verify(lbnd_as <= 0.0, "'lbnd_as' must not be positive; found %g", lbnd_as)
        _log.verify(
            ubnd_as - lbnd_as > 0.0,
            "'ubnd_as - lbnd_as' must be positive; found %g",
            ubnd_as - lbnd_as,
        )

        # hedging option
        # set strike == 0 to turn off
        strike = config(
            "strike", 1.0, float, help="Relative strike. Set to zero to turn off option"
        )
        ttm_steps = config(
            "ttm_steps", 4, int, help="Time to maturity of the option; in steps"
        )
        cost_v = config("cost_v", 0.02, float, help="Trading cost vega")
        cost_p = config(
            "cost_p",
            0.0005,
            float,
            help="Trading cost for the option on top of delta and vega cost",
        )
        ubnd_av = config(
            "ubnd_av",
            5.0,
            float,
            help="Upper bound for the number of options traded at each time step",
        )
        lbnd_av = config(
            "lbnd_av",
            -5.0,
            float,
            help="Lower bound for the number of options traded at each time step",
        )
        _log.verify(ttm_steps > 0, "'ttm_steps' must be positive; found %ld", ttm_steps)
        _log.verify(strike >= 0.0, "'strike' cannot be negative; found %g", strike)
        _log.verify(cost_v >= 0, "'cost_v' must not be negative; found %g", cost_v)
        _log.verify(cost_p >= 0, "'cost_p' must not be negative; found %g", cost_p)
        _log.verify(ubnd_av >= 0.0, "'ubnd_as' must not be negative; found %g", ubnd_av)
        _log.verify(lbnd_av <= 0.0, "'lbnd_av' must not be positive; found %g", lbnd_av)
        _log.verify(
            ubnd_av - lbnd_av > 0.0,
            "'ubnd_av - lbnd_as' must be positive; found %g",
            ubnd_av - lbnd_av,
        )

        # payoff
        # ------
        # must either be a function of spots[samples,steps+1], None, or a fixed umber
        payoff_f = config(
            "payoff",
            "atmcall",
            help="Payoff function with parameter spots[samples,steps+1]. Can be a function which must return a vector [samples]. Can also be short 'atmcall' or short 'atmput', or a fixed numnber. The default is 'atmcall' which is a short call with strike 1: '- np.maximum( spots[:,-1] - 1, 0. )'. A short forward starting ATM call is given as '- np.maximum( spots[:,-1] - spots[:,0], 0. )'.",
        )
        if payoff_f is None:
            # None means zero.
            payoff_f = np.zeros((nSamples,))
        elif isinstance(payoff_f, (int, float)):
            # specify terminal payoff as a fixed number, e.g. 0
            payoff_f = np.full((nSamples,), float(payoff_f))
        elif isinstance(payoff_f, str):
            if payoff_f == "atmcall":
                payoff_f = lambda spots: -np.maximum(spots[:, -1] - 1.0, 0.0)
            elif payoff_f == "atmput":
                payoff_f = lambda spots: -np.maximum(1.0 - spots[:, -1], 0.0)
            else:
                _log.throw("Unknown 'payoff' '%s'", payoff_f)

        # market dynamics
        # ---------------
        drift = config(
            "drift",
            0.1,
            float,
            help="Mean drift of the asset. This is the total drift.",
        )
        kappa_m = config(
            "meanrev_drift", 1.0, float, help="Mean reversion of the drift of the asset"
        )
        xi_m = config("drift_vol", 0.1, float, help="Vol of the drift")

        # vols
        rvol_init = config("rvol", 0.2, float, help="Initial realized volatility")
        ivol_init = config(
            "ivol",
            rvol_init,
            float,
            help="Initial implied volatility",
            help_default="Same as realized vol",
        )
        kappa_v = config(
            "meanrev_rvol",
            2.0,
            float,
            help="Mean reversion for realized vol vs implied vol",
        )
        kappa_i = config(
            "meanrev_ivol",
            0.1,
            float,
            help="Mean reversion for implied vol vol vs initial level",
        )
        xi_v = config("volvol_rvol", 0.5, float, help="Vol of Vol for realized vol")
        xi_i = config("volvol_ivol", 0.5, float, help="Vol of Vol for implied vol")
        _log.verify(rvol_init > 0.0, "'rvol' must be positive; found %", rvol_init)
        _log.verify(ivol_init > 0.0, "'ivol' must be positive; found %", ivol_init)

        # correlation
        rho_ms = config(
            "corr_ms", 0.5, float, help="Correlation between the asset and its mean"
        )
        rho_vs = config(
            "corr_vs",
            -0.7,
            float,
            help="Correlation between the asset and its volatility",
        )
        rho_vi = config(
            "corr_vi",
            0.8,
            float,
            help="Correlation between the implied vol and the asset volatility",
        )
        rho_vs_r = config(
            "rcorr_vs",
            -0.5,
            float,
            help="Residual correlation between the asset and its implied volatility",
        )

        _log.verify(
            abs(rho_ms) <= 1.0, "'rho_ms' must be between -1 and +1. Found %g", rho_ms
        )
        _log.verify(
            abs(rho_vs) <= 1.0, "'rho_vs' must be between -1 and +1. Found %g", rho_vs
        )
        _log.verify(
            abs(rho_vi) <= 1.0, "'rho_vi' must be between -1 and +1. Found %g", rho_vi
        )
        _log.verify(
            abs(rho_vs_r) <= 1.0,
            "'rho_vs_r' must be between -1 and +1. Found %g",
            rho_vs_r,
        )

        # close config
        config.done()
        self.usage_report = config.usage_report()
        self.input_report = config.input_report()

        # black scholes
        if bs_mode:
            strike = 0.0  # turn off option
            ttm_steps = 1
            nIvSteps = 0
            no_sdrift = True
            no_svol = True
        if no_sdrift:
            kappa_m = 0.0
            xi_m = 0.0
        if no_svol:
            kappa_v = 0.0
            kappa_i = 0.0
            xi_v = 0.0
            xi_i = 0.0

        # pre compute
        sqrtDt = math.sqrt(dt)
        ttm_steps = ttm_steps if strike > 0.0 else 1
        ttm = ttm_steps * dt
        sqrtTTM = math.sqrt(ttm)
        xi_m = abs(xi_m)  # negative number is odd, but forgivable
        xi_v = abs(xi_v)  # negative number is odd, but forgivable
        xi_i = abs(xi_i)  # negative number is odd, but forgivable
        time_left = (
            np.linspace(float(nSteps), 1.0, nSteps, endpoint=True, dtype=self.np_dtype)
            * dt
        )
        sqrt_time_left = np.sqrt(time_left)

        # simulate
        # --------
        # Not the most efficient simulator, but easier to read this way

        np.random.seed(seed)
        dW = (
            np.random.normal(
                size=(nSamples, nSteps + nIvSteps + ttm_steps - 1, 4)
            ).astype(self.np_dtype)
            * sqrtDt
        )
        dW_s = dW[:, :, 0]
        dW_m = dW[:, :, 0] * rho_ms + math.sqrt(1.0 - rho_ms**2) * dW[:, :, 1]
        dW_v = dW[:, :, 0] * rho_vs + math.sqrt(1.0 - rho_vs**2) * dW[:, :, 2]
        dW_i = dW[:, :, 2] * rho_vi + math.sqrt(1.0 - rho_vi**2) * (
            dW[:, :, 0] * rho_vs_r + math.sqrt(1.0 - rho_vs_r**2) * dW[:, :, 3]
        )

        spot = np.zeros((nSamples, nSteps + nIvSteps + ttm_steps), dtype=self.np_dtype)
        rdrift = np.full(
            (nSamples, nSteps + nIvSteps + ttm_steps), drift, dtype=self.np_dtype
        )
        rvol = np.full(
            (nSamples, nSteps + nIvSteps + ttm_steps), rvol_init, dtype=self.np_dtype
        )
        ivol = np.full(
            (nSamples, nSteps + nIvSteps + ttm_steps), ivol_init, dtype=self.np_dtype
        )

        spot[:, 0] = 1.0
        log_ivol_init = np.log(ivol_init)
        log_rvol = np.log(rvol_init)
        log_ivol = log_ivol_init
        rvol[:, 0] = rvol_init
        ivol[:, 0] = ivol_init
        mrdrift = 0.0 * dW_m[:, 0]
        expdriftdt = np.exp(drift * dt)
        bStochDrift = kappa_m != 0.0 or xi_m != 0.0
        bStochVol = kappa_v != 0.0 or xi_v != 0.0 or kappa_i != 0.0 or xi_i != 0.0

        for j in range(1, nSteps + nIvSteps + ttm_steps):
            # spot
            spot[:, j] = spot[:, j - 1] * np.exp(
                rdrift[:, j - 1] * dt
                + rvol[:, j - 1] * dW_s[:, j - 1]
                - 0.5 * (rvol[:, j - 1] ** 2) * dt
            )
            spot[:, j] *= expdriftdt / np.mean(spot[:, j])

            # drift
            # we normalize the stochastic drift to 'drift' on average.
            if bStochDrift:
                mrdrift = mrdrift - kappa_m * mrdrift * dt + xi_m * dW_m[:, j - 1]
                mrdrift = np.exp(mrdrift * dt)
                mrdrift = np.log(mrdrift / np.mean(mrdrift)) / dt
                rdrift[:, j] = drift + mrdrift

            # vols
            if bStochVol:
                log_rvol += (
                    kappa_v * (log_ivol - log_rvol) * dt
                    + xi_v * dW_v[:, j - 1]
                    - 0.5 * (xi_v**2) * dt
                )
                log_ivol += (
                    kappa_i * (log_ivol_init - log_ivol) * dt
                    + xi_i * dW_i[:, j - 1]
                    - 0.5 * (xi_i**2) * dt
                )
                rvol[:, j] = np.exp(log_rvol)
                ivol[:, j] = np.exp(log_ivol)

        # throw away the first nInvSteps
        # so we start in an invariant distribution
        spot = spot[:, nIvSteps:]
        rdrift = rdrift[:, nIvSteps : nIvSteps + nSteps]
        rvol = rvol[:, nIvSteps : nIvSteps + nSteps]
        ivol = ivol[:, nIvSteps : nIvSteps + nSteps]

        # sort
        ixs = np.argsort(spot[:, nSteps])
        spot = spot[ixs, :]
        rdrift = rdrift[ixs, :]
        rvol = rvol[ixs, :]
        ivol = ivol[ixs, :]

        # hedging instruments
        # -------------------

        dS = spot[:, nSteps][:, np.newaxis] - spot[:, :nSteps]
        cost_dS = spot[:, :nSteps] * cost_s
        if strike <= 0.0:
            dInsts = dS[:, :, np.newaxis]
            cost = cost_dS[:, :, np.newaxis]
            price = spot[:, :nSteps]
            ubnd_a = np.full((nSamples, nSteps, 1), ubnd_as)
            lbnd_a = np.full((nSamples, nSteps, 1), lbnd_as)

            call_price = None
            call_delta = None
            call_vega = None
            cost_dC = None

        else:
            # add hedging instrument: calls
            mat_spot = spot[
                :, ttm_steps : ttm_steps + nSteps
            ]  # spot at maturity of each option
            opt_spot = spot[:, :nSteps]  # spot at trading date of each option
            payoffs = np.maximum(0, mat_spot - strike * opt_spot)
            d1 = (-np.log(strike) + 0.5 * ivol * ivol * ttm) / (ivol * sqrtTTM)
            d2 = d1 - ivol * sqrtTTM
            N1 = norm.cdf(d1)
            N2 = norm.cdf(d2)
            call_price = N1 * opt_spot - N2 * strike * opt_spot
            dC = payoffs - call_price
            call_delta = N1
            call_vega = opt_spot * norm.pdf(d1) * sqrtTTM
            cost_dC = (
                cost_v * np.abs(call_vega)
                + cost_s * np.abs(call_delta)
                + cost_p * abs(call_price)
            )  # note: for a call vega and delta are positive, but we apply abs() anyway to illusteate the point

            dInsts = np.ones((nSamples, nSteps, 2), dtype=self.np_dtype)
            cost = np.ones((nSamples, nSteps, 2), dtype=self.np_dtype)
            price = np.ones((nSamples, nSteps, 2), dtype=self.np_dtype)
            ubnd_a = np.ones((nSamples, nSteps, 2), dtype=self.np_dtype)
            lbnd_a = np.ones((nSamples, nSteps, 2), dtype=self.np_dtype)
            dInsts[:, :, 0] = dS
            dInsts[:, :, 1] = dC
            cost[:, :, 0] = cost_dS
            cost[:, :, 1] = cost_dC
            price[:, :, 0] = spot[:, :nSteps]
            price[:, :, 1] = call_price
            ubnd_a[:, :, 0] = ubnd_as
            ubnd_a[:, :, 1] = ubnd_av
            lbnd_a[:, :, 0] = lbnd_as
            lbnd_a[:, :, 1] = lbnd_av

        # payoff
        # ------
        # The payoff function may return either payoff per sample, or dictionary with 'payoff' and 'features'
        # The features are expected to be of dimension (nSamples, nSteps, n).

        if not isinstance(payoff_f, np.ndarray):
            payoff = payoff_f(spot[:, : nSteps + 2])
            py_feat = None
            if isinstance(payoff, Mapping):
                py_feat = np.asarray(payoff["features"])
                payoff = np.asarray(payoff["payoff"])

                _log.verify(
                    len(py_feat.shape) in [2, 3],
                    "payoff['features']: must have dimension 2 or 3, found %ld",
                    len(py_feat.shape),
                )
                _log.verify(
                    py_feat.shape[0] == nSamples and py_feat.shape[1] == nSteps,
                    "payoff['features']: first two dimension must be (%ld,%ld). Found (%ld, %ld)",
                    nSamples,
                    nSteps,
                    py_feat.shape[0],
                    py_feat.shape[1],
                )
                py_feat = py_feat[:, 0] if len(py_feat) == 2 else py_feat
            else:
                payoff = np.asarray(payoff)

            payoff = payoff[:, 0] if payoff.shape == (nSamples, 1) else payoff
            _log.verify(
                payoff.shape == (nSamples,),
                "'payoff' function which receives a vector spots[nSamples,nSteps+1] must return a vector of size nSamples. Found shape %s",
                payoff.shape,
            )
        else:
            _log.verify(
                payoff_f.shape == (nSamples,),
                "'payoff' if a vector is provided, its size must match the sample size. Expected shape %s, found %s",
                (nSamples,),
                payoff_f.shape,
            )
            payoff = payoff_f
            py_feat = None

        # -----------------------------
        # unique_id
        # -----------------------------
        # Default handling for configs will ignore any function definitions, e.g. in this case 'payoff'.
        # we therefore manually generate a sufficient hash
        self.unique_id = uniqueHash(
            [config.input_dict(), payoff_f, self.tf_dtype.name], parse_functions=True
        )

        # -----------------------------
        # store data
        # -----------------------------

        # market
        # note that market variables are *not* automatically features
        # as they often look ahead

        self.data = pdct()
        self.data.market = pdct(
            hedges=dInsts, cost=cost, ubnd_a=ubnd_a, lbnd_a=lbnd_a, payoff=payoff
        )

        # features
        # observable variables for the agent
        self.data.features = pdct(
            per_step=pdct(
                # both spot and option, if present
                cost=cost,  # trading cost
                price=price,  # price
                ubnd_a=ubnd_a,  # bounds. Currently those are determinstic so don't use as features
                lbnd_a=lbnd_a,
                # time
                time_left=np.full(
                    (nSamples, nSteps), time_left[np.newaxis, :], dtype=self.np_dtype
                ),
                sqrt_time_left=np.full(
                    (nSamples, nSteps),
                    sqrt_time_left[np.newaxis, :],
                    dtype=self.np_dtype,
                ),
                # specific to equity spot
                spot=spot[
                    :, :nSteps
                ],  # spot level (S0,....,Sm-1). This does not include the terminal spot level.
                ivol=ivol,  # implied vol at beginning of each interval
            ),
            per_path=pdct(),
        )
        if strike > 0.0:
            self.data.features.per_step.update(
                call_price=call_price,  # price of the option
                call_delta=call_delta,  # delta
                call_vega=call_vega,  # vega
                cost_v=cost_dC,
            )

        if py_feat is not None:
            self.data.features.per_step.payoff_features = py_feat

        # the following variables must always be present in any world
        # it allows to cast dimensionless variables to the number of samples
        self.data.features.per_path[DIM_DUMMY] = (payoff * 0.0)[
            :, np.newaxis
        ]  # (None,1)

        # check numerics
        assert_iter_not_is_nan(self.data, "data")

        # data
        # what gym() gets

        self.tf_data = tf_dict(
            features=self.data.features, market=self.data.market, dtype=self.tf_dtype
        )

        # details
        # variables for visualization, but not available for the agent
        self.details = pdct(
            # mandatory (used by ploting)
            spot_all=spot[
                :, : nSteps + 1
            ],  # [nSamples,nSteps+1] spots including spot at T
            # per model
            drift=rdrift,  # drifts for each interval
            rvol=rvol,  # realized vols for each interval
        )

        # check numerics
        assert_iter_not_is_nan(self.details, "details")

        # generating sample weights
        # the tf_sample_weights is passed to keras train and must be of size [nSamples,1]
        # https://stackoverflow.com/questions/60399983/how-to-create-and-use-weighted-metrics-in-keras
        self.sample_weights = np.full(
            (nSamples, 1), 1.0 / float(nSamples), dtype=self.np_dtype
        )
        self.tf_sample_weights = tf.constant(
            self.sample_weights, dtype=self.tf_dtype
        )  # must be of size [nSamples,1] https://stackoverflow.com/questions/60399983/how-to-create-and-use-weighted-metrics-in-keras
        self.sample_weights = self.sample_weights.reshape((nSamples,))
        self.tf_y = tf.zeros((nSamples,), dtype=self.tf_dtype)
        self.nSteps = nSteps
        self.nSamples = nSamples
        self.nInst = 1 if strike <= 0.0 else 2
        self.dt = dt
        self.timeline = (
            np.cumsum(
                np.linspace(0.0, nSteps, nSteps + 1, endpoint=True, dtype=np.float32)
            )
            * dt
        )

        self.inst_names = ["spot"]
        if strike > 0.0:
            self.inst_names.append("ATM Call")

    def clone(self, config_overwrite=Config(), **kwargs):
        """
        Create a copy of this world with the same config, except for the seed.
        Used to generate genuine validation sets.

        Parameters
        ----------
            config_overwrite : Config, optional
                Allows specifying additional overwrites of specific config values
            **kwargs
                Allows specifying additional overwrites of specific config values, e.g.
                    world.clone( seed=222, samples=10 )
                If seed is not specified, a random seed is generated.

        Returns
        -------
            New world
        """
        if "seed" not in kwargs:
            kwargs["seed"] = int(np.random.randint(0, 0x7FFFFFFF))
        config = self.config.copy()
        config.update(config_overwrite, **kwargs)
        return SimpleWorld_Spot_ATM(config)

    def plot(self, config=Config(), **kwargs):
        """Plot simple world"""

        config.update(kwargs)
        col_size = config.fig("col_size", 5, int, "Figure column size")
        row_size = config.fig("row_size", 5, int, "Figure row size")
        plot_samples = config("plot_samples", 5, int, "Number of samples to plot")
        print_input = config(
            "print_input",
            True,
            bool,
            "Whether to print the config inputs for the world",
        )

        xSamples = np.linspace(
            0, self.nSamples, plot_samples, endpoint=False, dtype=int
        )
        timeline1 = (
            np.cumsum(
                np.linspace(
                    0.0, self.nSteps, self.nSteps + 1, endpoint=True, dtype=np.float32
                )
            )
            * self.dt
        )
        timeline = timeline1[:-1]

        print(self.config.usage_report())

        fig = figure(tight=True, col_size=col_size, row_size=row_size, col_nums=3)
        fig.suptitle(self.__class__.__name__, fontsize=16)

        # spot
        ax = fig.add_plot()
        ax.set_title("Spot")
        ax.set_xlabel("Time")
        for i, color in zip(xSamples, colors_tableau()):
            ax.plot(timeline1, self.details.spot_all[i, :], "-", color=color)
        ax.plot(
            timeline1,
            np.mean(self.details.spot_all, axis=0),
            "_",
            color="black",
            label="mean",
        )
        #        ax.get_xaxis().get_major_formatter().get_useOffset(False)
        ax.legend()

        # drift
        ax = fig.add_plot()
        ax.set_title("Drift")
        ax.set_xlabel("Time")
        for i, color in zip(xSamples, colors_tableau()):
            ax.plot(timeline, self.details.drift[i, :], "-", color=color)
        ax.plot(
            timeline,
            np.mean(self.details.drift, axis=0),
            "_",
            color="black",
            label="mean",
        )

        # vols
        ax = fig.add_plot()
        ax.set_title("Volatilities")
        ax.set_xlabel("Time")
        for i, color in zip(xSamples, colors_tableau()):
            ax.plot(timeline, self.data.features.per_step.ivol[i, :], "-", color=color)
            ax.plot(timeline, self.details.rvol[i, :], ":", color=color)

        if self.nInst > 1:
            # call prices
            ax = fig.add_plot(True)
            ax.set_title("Call Prices")
            ax.set_xlabel("Time")
            for i, color in zip(xSamples, colors_tableau()):
                ax.plot(
                    timeline,
                    self.data.features.per_step.call_price[i, :],
                    "-",
                    color=color,
                )
            ax.plot(
                timeline,
                np.mean(self.data.features.per_step.call_price, axis=0),
                "_",
                color="black",
                label="mean",
            )
            ax.legend()

            # call delta
            ax = fig.add_plot()
            ax.set_title("Call Deltas")
            ax.set_xlabel("Time")
            for i, color in zip(xSamples, colors_tableau()):
                ax.plot(
                    timeline,
                    self.data.features.per_step.call_delta[i, :],
                    "-",
                    color=color,
                )
            ax.plot(
                timeline,
                np.mean(self.data.features.per_step.call_delta, axis=0),
                "_",
                color="black",
                label="mean",
            )
            ax.legend()

            # call vega
            ax = fig.add_plot()
            ax.set_title("Call Vegas")
            ax.set_xlabel("Time")
            for i, color in zip(xSamples, colors_tableau()):
                ax.plot(
                    timeline,
                    self.data.features.per_step.call_vega[i, :],
                    "-",
                    color=color,
                )
            ax.plot(
                timeline,
                np.mean(self.data.features.per_step.call_vega, axis=0),
                "_",
                color="black",
                label="mean",
            )
            ax.legend()

        fig.render()
        del fig

        if print_input:
            print("Config settings:\n%s" % self.input_report)


class PPAWorld(object):
    def _store_clean_world(
        self,
        *,
        config,
        nSamples,
        nSteps,
        dt,
        payoff,
        dInsts,
        price,
        cost,
        ubnd_a,
        lbnd_a,
        ubnd_trade,
        lbnd_trade,
        per_step_features,
        details,
        latent_model_type,
        feature_model_type,
        ppa_contract,
        use_transaction_cost,
        inst_names,
        site_country_codes=None,
        site_type_codes=None,
        site_country_names=None,
        site_type_names=None,
    ):
        self.unique_id = uniqueHash([config.input_dict(), self.tf_dtype.name])

        self.data = pdct()
        self.data.market = pdct(
            hedges=dInsts,
            cost=cost,
            ubnd_a=ubnd_a,
            lbnd_a=lbnd_a,
            ubnd_trade=ubnd_trade,
            lbnd_trade=lbnd_trade,
            payoff=payoff,
        )
        self.data.features = pdct(per_step=per_step_features, per_path=pdct())
        self.data.features.per_path[DIM_DUMMY] = (payoff * 0.0)[:, np.newaxis]

        assert_iter_not_is_nan(self.data, "data")
        self.tf_data = tf_dict(
            features=self.data.features, market=self.data.market, dtype=self.tf_dtype
        )

        self.details = details
        assert_iter_not_is_nan(self.details, "details")

        self.sample_weights = np.full(
            (nSamples, 1), 1.0 / float(nSamples), dtype=self.np_dtype
        )
        self.tf_sample_weights = tf.constant(
            self.sample_weights, dtype=self.tf_dtype
        )
        self.sample_weights = self.sample_weights.reshape((nSamples,))
        self.tf_y = tf.zeros((nSamples,), dtype=self.tf_dtype)

        self.nSteps = nSteps
        self.nSamples = nSamples
        self.nInst = dInsts.shape[-1]
        self.dt = dt
        self.latent_model_type = latent_model_type
        self.feature_model_type = feature_model_type
        self.ppa_contract = ppa_contract
        self.use_transaction_cost = use_transaction_cost
        self.timeline = (
            np.cumsum(
                np.linspace(0.0, nSteps, nSteps + 1, endpoint=True, dtype=np.float32)
            )
            * dt
        )
        self.inst_names = inst_names
        self.site_country_codes = site_country_codes
        self.site_type_codes = site_type_codes
        self.site_country_names = site_country_names
        self.site_type_names = site_type_names

    def _expected_sigmoid_normal(self, mean, var, phi, gh_x, gh_w):
        mean = np.asarray(mean, dtype=self.np_dtype)
        nodes = mean[:, np.newaxis] + phi + np.sqrt(2.0 * var) * gh_x
        sigmoid_nodes = 1.0 / (1.0 + np.exp(-np.clip(nodes, -60.0, 60.0)))
        return np.sum(gh_w * sigmoid_nodes, axis=1) / np.sqrt(
            np.pi
        )

    def _calibrate_sigmoid_phi(self, target, kappa, sigma, horizon, gh_x, gh_w):
        var = (sigma**2 / (2.0 * kappa)) * (1.0 - np.exp(-2.0 * kappa * horizon))

        def objective(phi):
            return (
                self._expected_sigmoid_normal(
                    np.array([0.0], dtype=self.np_dtype), var, phi, gh_x, gh_w
                )[0]
                - target
            )

        return bisect(objective, -8.0, 8.0)

    def _simulate_clean_site_forecasts(
        self,
        *,
        rng,
        nSamples,
        nSteps,
        dt,
        horizon,
        n_sites,
        target,
        kappa,
        sigma,
        corr_matrix,
    ):
        gh_x, gh_w = np.polynomial.hermite.hermgauss(40)
        gh_x = gh_x.astype(self.np_dtype)
        gh_w = gh_w.astype(self.np_dtype)
        phi = self._calibrate_sigmoid_phi(target, kappa, sigma, horizon, gh_x, gh_w)
        phis = np.full(n_sites, phi, dtype=self.np_dtype)

        persistence = np.exp(-kappa * dt)
        step_scale = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))
        chol = np.linalg.cholesky(corr_matrix).astype(self.np_dtype)

        x = np.zeros((nSamples, nSteps + 1, n_sites), dtype=self.np_dtype)
        q_forecasts = np.zeros_like(x)
        for t in range(nSteps + 1):
            if t > 0:
                z = rng.normal(size=(nSamples, n_sites)).astype(self.np_dtype)
                x[:, t, :] = persistence * x[:, t - 1, :] + step_scale * (z @ chol.T)
            time_to_delivery = horizon - (t * dt)
            if time_to_delivery <= 0.0:
                q_forecasts[:, t, :] = 1.0 / (
                    1.0 + np.exp(-np.clip(x[:, t, :] + phis, -60.0, 60.0))
                )
            else:
                var = (sigma**2 / (2.0 * kappa)) * (
                    1.0 - np.exp(-2.0 * kappa * time_to_delivery)
                )
                mean = x[:, t, :] * np.exp(-kappa * time_to_delivery)
                for i in range(n_sites):
                    q_forecasts[:, t, i] = self._expected_sigmoid_normal(
                        mean[:, i], var, phis[i], gh_x, gh_w
                    )

        q_realized_sites = q_forecasts[:, -1, :]
        return q_forecasts, q_realized_sites

    def _two_region_corr(self, n_sites, split, within_corr, cross_corr):
        corr = np.full((n_sites, n_sites), cross_corr, dtype=self.np_dtype)
        corr[:split, :split] = within_corr
        corr[split:, split:] = within_corr
        np.fill_diagonal(corr, 1.0)
        min_eig = float(np.min(np.linalg.eigvalsh(corr)))
        if min_eig <= 1e-8:
            corr += np.eye(n_sites, dtype=self.np_dtype) * (1e-8 - min_eig)
        return corr

    def _price_factor(self, *, rng, nSamples, nSteps, dt, kappa, sigma):
        x = np.zeros((nSamples, nSteps + 1), dtype=self.np_dtype)
        persistence = np.exp(-kappa * dt)
        step_scale = sigma * np.sqrt((1.0 - np.exp(-2.0 * kappa * dt)) / (2.0 * kappa))
        for t in range(1, nSteps + 1):
            x[:, t] = persistence * x[:, t - 1] + step_scale * rng.normal(size=nSamples)
        return x

    def _clean_bounds_and_costs(
        self,
        *,
        nSamples,
        nSteps,
        nInst,
        price,
        ubnd_a_value,
        lbnd_a_value,
        ubnd_trade_value,
        lbnd_trade_value,
        use_transaction_cost,
    ):
        if price.ndim == 2:
            price_for_cost = price[:, :nSteps, np.newaxis]
        else:
            price_for_cost = price[:, :nSteps, :]

        spread_base = 0.01
        spread_max = 0.03
        tau = np.linspace(float(nSteps), 1.0, nSteps, endpoint=True, dtype=self.np_dtype)
        lamb = 5.0
        bid_ask_spread_free = spread_base + spread_max * np.exp(-lamb * tau)
        cost_matrix = price_for_cost * (bid_ask_spread_free[np.newaxis, :, np.newaxis] / 2.0) + 0.15
        cost = (
            cost_matrix.astype(self.np_dtype, copy=False)
            if use_transaction_cost
            else np.zeros((nSamples, nSteps, nInst), dtype=self.np_dtype)
        )
        ubnd_a = np.full((nSamples, nSteps, nInst), ubnd_a_value, dtype=self.np_dtype)
        lbnd_a = np.full((nSamples, nSteps, nInst), lbnd_a_value, dtype=self.np_dtype)
        ubnd_trade = np.full(
            (nSamples, nSteps, nInst), ubnd_trade_value, dtype=self.np_dtype
        )
        lbnd_trade = np.full(
            (nSamples, nSteps, nInst), lbnd_trade_value, dtype=self.np_dtype
        )
        return cost, ubnd_a, lbnd_a, ubnd_trade, lbnd_trade

    def _init_biegler_koenig_replication_world(self, config: Config):
        """
        Clean Biegler-Koenig replication world.

        This constructor keeps the original paper-style replication separate from
        the congestion experiments:
        - two OU wind factors: onshore and offshore;
        - one national forward price driven by the weighted aggregate expected
          wind infeed;
        - one pay-as-produced PPA liability written on the onshore wind asset;
        - no congestion, no delivered-volume adjustment.
        """
        nSamples = config("samples", 100000, int, help="Number of simulated paths")
        nSteps = config("steps", 48, int, help="Number of time steps")
        seed = config("seed", 2312414312, int, help="Random seed")
        horizon = config("max_time", 48.0, float, help="Delivery horizon")
        dt = config("dt_replicate", horizon / nSteps, float, help="Time step")

        kappa_q1 = config(
            "mean_rev_q1", 0.1 / 24.0, float, help="Mean reversion onshore"
        )
        sigma_q1 = config(
            "vol_q1", 3.0 / np.sqrt(365.0 * 24.0), float, help="Volatility onshore"
        )
        kappa_q2 = config(
            "mean_rev_q2", 0.1 / 24.0, float, help="Mean reversion offshore"
        )
        sigma_q2 = config(
            "vol_q2", 3.0 / np.sqrt(365.0 * 24.0), float, help="Volatility offshore"
        )
        kappa_p = config("mean_rev_p", 0.5 / 24.0, float, help="Price mean reversion")
        sigma_p = config(
            "vol_p", 0.8 / np.sqrt(365.0 * 24.0), float, help="Price volatility"
        )
        rho = config("rho", 0.46, float, help="Onshore/offshore wind correlation")
        q1_target = config("q1_target", 0.5, float, help="Initial onshore forecast")
        q2_target = config("q2_target", 0.5, float, help="Initial offshore forecast")
        w1 = config("w1", 0.8, float, help="Onshore national wind weight")
        w2 = config("w2", 0.2, float, help="Offshore national wind weight")
        f_0_T = config("f_0_T", 100.0, float, help="Initial forward price")
        ppa_strike = config("ppa_strike", 100.0, float, help="PPA strike")
        feature_model_type = config(
            "feature_model_type", "A", str, help="Information set: A, B, or C"
        ).strip().upper()
        use_transaction_cost = config(
            "use_transaction_cost", False, bool, help="Include transaction costs"
        )
        ubnd_a_value = config("ubnd_a", 5.0, float, help="Inventory upper bound")
        lbnd_a_value = config("lbnd_a", -5.0, float, help="Inventory lower bound")
        ubnd_trade_value = config("ubnd_trade", 1.0e6, float, help="Trade upper bound")
        lbnd_trade_value = config("lbnd_trade", -1.0e6, float, help="Trade lower bound")

        _log.verify(nSteps > 0, "'steps' must be positive; found %d", nSteps)
        _log.verify(nSamples > 0, "'samples' must be positive; found %d", nSamples)
        _log.verify(dt > 0.0, "'dt_replicate' must be positive; found %g", dt)
        _log.verify(ubnd_a_value >= 0.0, "'ubnd_a' must not be negative; found %g", ubnd_a_value)
        _log.verify(lbnd_a_value <= 0.0, "'lbnd_a' must not be positive; found %g", lbnd_a_value)
        _log.verify(ubnd_trade_value >= 0.0, "'ubnd_trade' must not be negative; found %g", ubnd_trade_value)
        _log.verify(lbnd_trade_value <= 0.0, "'lbnd_trade' must not be positive; found %g", lbnd_trade_value)

        weights = np.array([w1, w2], dtype=self.np_dtype)
        weight_sum = float(np.sum(weights))
        _log.verify(weight_sum > 0.0, "'w1 + w2' must be positive; found %g", weight_sum)
        weights = weights / weight_sum
        k_vals = np.array([kappa_q1, kappa_q2], dtype=self.np_dtype)
        s_vals = np.array([sigma_q1, sigma_q2], dtype=self.np_dtype)
        targets = np.array([q1_target, q2_target], dtype=self.np_dtype)

        gh_x, gh_w = np.polynomial.hermite.hermgauss(40)
        gh_x = gh_x.astype(self.np_dtype)
        gh_w = gh_w.astype(self.np_dtype)
        phis = np.array(
            [
                self._calibrate_sigmoid_phi(
                    float(targets[i]), float(k_vals[i]), float(s_vals[i]), horizon, gh_x, gh_w
                )
                for i in range(2)
            ],
            dtype=self.np_dtype,
        )

        rng = np.random.RandomState(seed)
        x = np.zeros((nSamples, nSteps + 1, 2), dtype=self.np_dtype)
        q_forecasts = np.zeros_like(x)
        xp = np.zeros((nSamples, nSteps + 1), dtype=self.np_dtype)

        persistence = np.exp(-k_vals * dt).astype(self.np_dtype)
        cov_matrix = np.array(
            [
                [sigma_q1**2, rho * sigma_q1 * sigma_q2],
                [rho * sigma_q1 * sigma_q2, sigma_q2**2],
            ],
            dtype=self.np_dtype,
        )
        # Keep the old replication discretization convention, where both
        # factors share the q1 time-integral scale.
        discrete_cov = cov_matrix * (
            (1.0 - np.exp(-2.0 * kappa_q1 * dt)) / (2.0 * kappa_q1)
        )
        vol_step = np.linalg.cholesky(discrete_cov).astype(self.np_dtype)
        price_persistence = np.exp(-kappa_p * dt)
        price_step = sigma_p * np.sqrt(
            (1.0 - np.exp(-2.0 * kappa_p * dt)) / (2.0 * kappa_p)
        )

        for t in range(nSteps + 1):
            if t > 0:
                z = rng.normal(size=(nSamples, 2)).astype(self.np_dtype)
                x[:, t, :] = x[:, t - 1, :] * persistence[np.newaxis, :] + z @ vol_step.T
                zp = rng.normal(size=nSamples).astype(self.np_dtype)
                xp[:, t] = price_persistence * xp[:, t - 1] + price_step * zp

            time_to_delivery = horizon - (t * dt)
            for i in range(2):
                if time_to_delivery <= 0.0:
                    q_forecasts[:, t, i] = 1.0 / (
                        1.0 + np.exp(-np.clip(x[:, t, i] + phis[i], -60.0, 60.0))
                    )
                else:
                    mean = x[:, t, i] * np.exp(-k_vals[i] * time_to_delivery)
                    var = (s_vals[i] ** 2 / (2.0 * k_vals[i])) * (
                        1.0 - np.exp(-2.0 * k_vals[i] * time_to_delivery)
                    )
                    q_forecasts[:, t, i] = self._expected_sigmoid_normal(
                        mean, var, phis[i], gh_x, gh_w
                    )

        q_realized_sites = q_forecasts[:, -1, :]
        q_agg_forecast = np.sum(weights[np.newaxis, np.newaxis, :] * q_forecasts, axis=2)
        q_total_realized = np.sum(weights[np.newaxis, :] * q_realized_sites, axis=1)
        q_dispersion = np.sqrt(
            np.sum(
                weights[np.newaxis, np.newaxis, :]
                * (q_forecasts - q_agg_forecast[:, :, np.newaxis]) ** 2,
                axis=2,
            )
        )

        g0 = 1.0 - float(q_agg_forecast[0, 0])
        f_t_T = np.zeros((nSamples, nSteps + 1), dtype=self.np_dtype)
        for t in range(nSteps + 1):
            time_to_delivery = horizon - (t * dt)
            mt = xp[:, t] * np.exp(-kappa_p * time_to_delivery)
            gt = 1.0 - q_agg_forecast[:, t]
            f_t_T[:, t] = f_0_T * (1.0 + mt) * (gt / g0)

        q_ppa = q_realized_sites[:, 0]
        payoff = q_ppa * (f_t_T[:, -1] - ppa_strike)
        dInsts = (f_t_T[:, -1, np.newaxis] - f_t_T[:, :nSteps])[:, :, np.newaxis]
        price = f_t_T[:, :nSteps]
        cost, ubnd_a, lbnd_a, ubnd_trade, lbnd_trade = self._clean_bounds_and_costs(
            nSamples=nSamples,
            nSteps=nSteps,
            nInst=1,
            price=price,
            ubnd_a_value=ubnd_a_value,
            lbnd_a_value=lbnd_a_value,
            ubnd_trade_value=ubnd_trade_value,
            lbnd_trade_value=lbnd_trade_value,
            use_transaction_cost=use_transaction_cost,
        )

        time_left = (
            np.linspace(float(nSteps), 1.0, nSteps, endpoint=True, dtype=self.np_dtype)
            * dt
        )
        per_step_features = pdct(
            time_left=np.full((nSamples, nSteps), time_left[np.newaxis, :], dtype=self.np_dtype),
            forward_price=price,
            cost=cost,
        )
        if feature_model_type == "A":
            per_step_features.wind_info = q_agg_forecast[:, :-1, np.newaxis]
        elif feature_model_type == "B":
            per_step_features.wind_info = np.stack(
                [
                    weights[0] * q_forecasts[:, :-1, 0],
                    weights[1] * q_forecasts[:, :-1, 1],
                ],
                axis=-1,
            )
        elif feature_model_type == "C":
            per_step_features.wind_info = q_forecasts[:, :-1, :]
        else:
            raise ValueError("feature_model_type must be one of {'A', 'B', 'C'}.")

        path_history = np.stack([f_t_T, q_forecasts[:, :, 0], q_agg_forecast], axis=-1)
        details = pdct(
            forward_price=f_t_T,
            onshore_wind=q_forecasts[:, :, 0],
            offshore_wind=q_forecasts[:, :, 1],
            q1_forecast=q_forecasts[:, :, 0],
            q2_forecast=q_forecasts[:, :, 1],
            q_agg_forecast=q_agg_forecast,
            q_cong_forecast=weights[0] * q_forecasts[:, :, 0],
            q_unc_forecast=weights[1] * q_forecasts[:, :, 1],
            q_cross_sectional_dispersion_forecast=q_dispersion,
            q_total_realized=q_total_realized,
            q1_realized=q_realized_sites[:, 0],
            q2_realized=q_realized_sites[:, 1],
            q_tilde=q_ppa,
            q_cong=weights[0] * q_realized_sites[:, 0],
            q_unc=weights[1] * q_realized_sites[:, 1],
            q_cong_flow=weights[0] * q_realized_sites[:, 0],
            q_curtailment=np.zeros(nSamples, dtype=self.np_dtype),
            q_curtailment_ratio=np.zeros(nSamples, dtype=self.np_dtype),
            q_cluster_realized=q_ppa,
            q_cluster_delivered=q_ppa,
            q_congestion_region_realized=np.zeros(nSamples, dtype=self.np_dtype),
            q_congestion_flow=np.zeros(nSamples, dtype=self.np_dtype),
            q_congestion_curtailment_ratio=np.zeros(nSamples, dtype=self.np_dtype),
            q_cong_ratio_realized=np.divide(
                weights[0] * q_realized_sites[:, 0],
                q_total_realized,
                out=np.zeros(nSamples, dtype=self.np_dtype),
                where=q_total_realized > 1e-12,
            ),
            payoff=payoff,
            path_history=path_history,
            cannibal_rat=np.mean(f_t_T[:, -1, np.newaxis] * q_realized_sites, axis=0)
            / np.mean(q_realized_sites, axis=0)
            / f_0_T,
            site_forecasts=q_forecasts,
            site_realized=q_realized_sites,
            ppa_cluster_indices=np.array([0], dtype=int),
            ppa_contract_weights=np.array([1.0], dtype=self.np_dtype),
            ppa_capacity_scale=np.array(1.0),
            congestion_region_indices=np.zeros(0, dtype=int),
            congestion_flow_weights=np.zeros(0, dtype=self.np_dtype),
            l_max=np.array(np.inf, dtype=self.np_dtype),
        )
        self._store_clean_world(
            config=config,
            nSamples=nSamples,
            nSteps=nSteps,
            dt=dt,
            payoff=payoff,
            dInsts=dInsts,
            price=price,
            cost=cost,
            ubnd_a=ubnd_a,
            lbnd_a=lbnd_a,
            ubnd_trade=ubnd_trade,
            lbnd_trade=lbnd_trade,
            per_step_features=per_step_features,
            details=details,
            latent_model_type="biegler_koenig_replication",
            feature_model_type=feature_model_type,
            ppa_contract="onshore_pay_as_produced",
            use_transaction_cost=use_transaction_cost,
            inst_names=["Forward"],
            site_country_codes=np.zeros(2, dtype=int),
            site_type_codes=np.array([0, 1], dtype=int),
            site_country_names=["Market"],
            site_type_names=["onshore", "offshore"],
        )

    def _init_spatial_volume_risk_world(self, config: Config):
        """
        Minimal one-country volume-risk experiment.

        This is the clean spatial volume-risk mechanism:
        - the traded forward price is national and uses the Biegler-König-style
          uncurtailed aggregate wind forecast;
        - the PPA volume is physically delivered production;
        - the first region can be curtailed through min(Q_cong, L_max);
        - A sees aggregate wind, B sees aggregate wind plus dispersion, and C sees
          all site-level forecasts.
        """
        nSamples = config("samples", 100000, int, help="Number of simulated paths")
        nSteps = config("steps", 48, int, help="Number of time steps")
        seed = config("seed", 2312414312, int, help="Random seed")
        horizon = config("max_time", 48.0, float, help="Delivery horizon")
        dt = config("dt_replicate", horizon / nSteps, float, help="Time step")
        n_sites = config("synthetic_num_sites", 10, int, help="Number of wind sites")
        split = config(
            "synthetic_minimal_congested_sites",
            n_sites // 2,
            int,
            help="Number of sites in the congested region",
        )
        target = config("synthetic_target", 0.5, float, help="Wind target")
        kappa = config("synthetic_mean_reversion", 0.02, float, help="Wind mean reversion")
        sigma = config("synthetic_vol", 4.0, float, help="Wind volatility")
        within_corr = config(
            "synthetic_two_region_within_corr",
            0.90,
            float,
            help="Within-region wind correlation",
        )
        cross_corr = config(
            "synthetic_two_region_cross_corr",
            -0.50,
            float,
            help="Cross-region wind correlation",
        )
        f_0_T = config("f_0_T", 100.0, float, help="Initial forward price")
        ppa_strike = config("ppa_strike", f_0_T, float, help="PPA strike")
        l_max = config("l_max", np.inf, float, help="Congested export limit")
        kappa_p = config("mean_rev_p", 0.5 / 24.0, float, help="Price mean reversion")
        sigma_p = config(
            "vol_p", 0.25 / np.sqrt(365.0 * 24.0), float, help="Price-factor volatility"
        )
        feature_model_type = config(
            "feature_model_type", "A", str, help="Information set: A, B, or C"
        ).strip().upper()
        use_transaction_cost = config(
            "use_transaction_cost", False, bool, help="Include transaction costs"
        )
        ubnd_a_value = config("ubnd_a", 1.5, float, help="Inventory upper bound")
        lbnd_a_value = config("lbnd_a", -1.5, float, help="Inventory lower bound")
        ubnd_trade_value = config("ubnd_trade", 5.0, float, help="Trade upper bound")
        lbnd_trade_value = config("lbnd_trade", -5.0, float, help="Trade lower bound")

        _log.verify(n_sites % 2 == 0, "spatial_volume_risk requires an even site count.")
        _log.verify(0 < split < n_sites, "Invalid congested split %d for %d sites.", split, n_sites)

        rng = np.random.default_rng(seed)
        corr = self._two_region_corr(n_sites, split, within_corr, cross_corr)
        q_forecasts, q_realized_sites = self._simulate_clean_site_forecasts(
            rng=rng,
            nSamples=nSamples,
            nSteps=nSteps,
            dt=dt,
            horizon=horizon,
            n_sites=n_sites,
            target=target,
            kappa=kappa,
            sigma=sigma,
            corr_matrix=corr,
        )
        weights = np.full(n_sites, 1.0 / n_sites, dtype=self.np_dtype)
        q_agg_forecast = np.sum(weights * q_forecasts, axis=2)
        q_total_realized = np.sum(weights * q_realized_sites, axis=1)
        q_cong_forecast = np.sum(weights[:split] * q_forecasts[:, :, :split], axis=2)
        q_unc_forecast = np.sum(weights[split:] * q_forecasts[:, :, split:], axis=2)
        q_cong = np.sum(weights[:split] * q_realized_sites[:, :split], axis=1)
        q_unc = np.sum(weights[split:] * q_realized_sites[:, split:], axis=1)
        q_tilde = np.minimum(q_cong, l_max) + q_unc
        q_dispersion = np.sqrt(
            np.sum(weights[np.newaxis, np.newaxis, :] * (q_forecasts - q_agg_forecast[:, :, np.newaxis]) ** 2, axis=2)
        )

        xp = self._price_factor(
            rng=rng,
            nSamples=nSamples,
            nSteps=nSteps,
            dt=dt,
            kappa=kappa_p,
            sigma=sigma_p,
        )
        g0 = 1.0 - float(np.mean(q_agg_forecast[:, 0]))
        f_t_T = np.zeros((nSamples, nSteps + 1), dtype=self.np_dtype)
        for t in range(nSteps + 1):
            time_to_delivery = horizon - (t * dt)
            mt = xp[:, t] * np.exp(-kappa_p * time_to_delivery)
            gt = 1.0 - q_agg_forecast[:, t]
            f_t_T[:, t] = f_0_T * (1.0 + mt) * (gt / g0)

        payoff = q_tilde * (f_t_T[:, -1] - ppa_strike)
        dInsts = (f_t_T[:, -1, np.newaxis] - f_t_T[:, :nSteps])[:, :, np.newaxis]
        price = f_t_T[:, :nSteps]
        cost, ubnd_a, lbnd_a, ubnd_trade, lbnd_trade = self._clean_bounds_and_costs(
            nSamples=nSamples,
            nSteps=nSteps,
            nInst=1,
            price=price,
            ubnd_a_value=ubnd_a_value,
            lbnd_a_value=lbnd_a_value,
            ubnd_trade_value=ubnd_trade_value,
            lbnd_trade_value=lbnd_trade_value,
            use_transaction_cost=use_transaction_cost,
        )

        per_step_features = pdct(
            time_left=np.full(
                (nSamples, nSteps),
                np.linspace(float(nSteps), 1.0, nSteps, endpoint=True, dtype=self.np_dtype)[np.newaxis, :] * dt,
                dtype=self.np_dtype,
            ),
            forward_price=price,
            cost=cost,
        )
        if feature_model_type == "A":
            per_step_features.wind_info = q_agg_forecast[:, :-1, np.newaxis]
        elif feature_model_type == "B":
            per_step_features.wind_info = np.stack(
                [q_agg_forecast[:, :-1], q_dispersion[:, :-1]], axis=-1
            )
        elif feature_model_type == "C":
            per_step_features.wind_info = q_forecasts[:, :-1, :]
        else:
            raise ValueError("feature_model_type must be one of {'A', 'B', 'C'}.")

        curtailment = q_total_realized - q_tilde
        details = pdct(
            forward_price=f_t_T,
            q_agg_forecast=q_agg_forecast,
            q_cong_forecast=q_cong_forecast,
            q_unc_forecast=q_unc_forecast,
            q_cross_sectional_dispersion_forecast=q_dispersion,
            q_total_realized=q_total_realized,
            q_cong=q_cong,
            q_unc=q_unc,
            q_cong_flow=q_cong,
            q_tilde=q_tilde,
            q_curtailment=curtailment,
            q_curtailment_ratio=np.divide(
                curtailment,
                q_total_realized,
                out=np.zeros_like(curtailment),
                where=q_total_realized > 1e-12,
            ),
            q_cluster_realized=q_total_realized,
            q_cluster_delivered=q_tilde,
            q_congestion_region_realized=q_cong,
            q_congestion_flow=q_cong,
            q_congestion_curtailment_ratio=np.divide(
                q_cong - np.minimum(q_cong, l_max),
                q_cong,
                out=np.zeros_like(q_cong),
                where=q_cong > 1e-12,
            ),
            payoff=payoff,
            path_history=np.stack([f_t_T, q_agg_forecast, q_cong_forecast], axis=-1),
            cannibal_rat=np.mean(f_t_T[:, -1, np.newaxis] * q_realized_sites, axis=0)
            / np.mean(q_realized_sites, axis=0)
            / f_0_T,
            site_forecasts=q_forecasts,
            site_realized=q_realized_sites,
            ppa_cluster_indices=np.arange(n_sites, dtype=int),
            ppa_contract_weights=weights,
            ppa_capacity_scale=np.array(1.0),
            congestion_region_indices=np.arange(split, dtype=int),
            congestion_flow_weights=np.ones(split, dtype=self.np_dtype),
            q_cong_ratio_realized=np.divide(
                q_cong,
                q_total_realized,
                out=np.zeros_like(q_cong),
                where=q_total_realized > 1e-12,
            ),
            l_max=np.array(l_max, dtype=self.np_dtype),
        )
        site_country_codes = np.zeros(n_sites, dtype=int)
        site_type_codes = np.array([1] * split + [0] * (n_sites - split), dtype=int)
        self._store_clean_world(
            config=config,
            nSamples=nSamples,
            nSteps=nSteps,
            dt=dt,
            payoff=payoff,
            dInsts=dInsts,
            price=price,
            cost=cost,
            ubnd_a=ubnd_a,
            lbnd_a=lbnd_a,
            ubnd_trade=ubnd_trade,
            lbnd_trade=lbnd_trade,
            per_step_features=per_step_features,
            details=details,
            latent_model_type="spatial_volume_risk",
            feature_model_type=feature_model_type,
            ppa_contract="regional_volume_risk",
            use_transaction_cost=use_transaction_cost,
            inst_names=["DE Forward"],
            site_country_codes=site_country_codes,
            site_type_codes=site_type_codes,
            site_country_names=["DE"],
            site_type_names=["unconstrained", "congested"],
        )

    def _init_simple_cross_border_extension_world(self, config: Config):
        """
        Lightweight cross-border extension.

        The German local PPA remains the liability. The German forward is the
        natural hedge, while the Dutch forward is only an additional instrument.
        Prices are country-specific national forwards; congestion only affects
        delivered German PPA volume.
        """
        nSamples = config("samples", 100000, int, help="Number of simulated paths")
        nSteps = config("steps", 48, int, help="Number of time steps")
        seed = config("seed", 2312414312, int, help="Random seed")
        horizon = config("max_time", 48.0, float, help="Delivery horizon")
        dt = config("dt_replicate", horizon / nSteps, float, help="Time step")
        sites_per_country = config(
            "synthetic_sites_per_country", 10, int, help="Sites per country"
        )
        split = config(
            "synthetic_minimal_congested_sites",
            sites_per_country // 2,
            int,
            help="German congested sites",
        )
        target = config("synthetic_target", 0.5, float, help="Wind target")
        kappa = config("synthetic_mean_reversion", 0.02, float, help="Wind mean reversion")
        sigma = config("synthetic_vol", 4.0, float, help="Wind volatility")
        within_corr = config("synthetic_two_region_within_corr", 0.85, float, help="Within-block correlation")
        cross_country_corr = config("synthetic_cross_country_corr", 0.30, float, help="DE/NL wind correlation")
        f_0_T_de = config("f_0_T_de", 100.0, float, help="Initial DE forward")
        f_0_T_nl = config("f_0_T_nl", 100.0, float, help="Initial NL forward")
        ppa_strike_de = config("ppa_strike_de", f_0_T_de, float, help="German PPA strike")
        l_max_default = config("l_max", np.inf, float, help="German export limit")
        l_max_de = config("l_max_de", l_max_default, float, help="German export limit")
        kappa_p = config("mean_rev_p", 0.5 / 24.0, float, help="Price mean reversion")
        sigma_p = config("vol_p", 0.25 / np.sqrt(365.0 * 24.0), float, help="Price volatility")
        rho_price = config("rho_price_cross_border", 0.40, float, help="DE/NL price-factor correlation")
        feature_model_type = config("feature_model_type", "A", str, help="Information set").strip().upper()
        use_transaction_cost = config("use_transaction_cost", False, bool, help="Include transaction costs")
        ubnd_a_value = config("ubnd_a", 1.5, float, help="Inventory upper bound")
        lbnd_a_value = config("lbnd_a", -1.5, float, help="Inventory lower bound")
        ubnd_trade_value = config("ubnd_trade", 5.0, float, help="Trade upper bound")
        lbnd_trade_value = config("lbnd_trade", -5.0, float, help="Trade lower bound")

        n_sites = 2 * sites_per_country
        rng = np.random.default_rng(seed)
        corr = np.full((n_sites, n_sites), cross_country_corr, dtype=self.np_dtype)
        corr[:sites_per_country, :sites_per_country] = within_corr
        corr[sites_per_country:, sites_per_country:] = within_corr
        np.fill_diagonal(corr, 1.0)
        min_eig = float(np.min(np.linalg.eigvalsh(corr)))
        if min_eig <= 1e-8:
            corr += np.eye(n_sites, dtype=self.np_dtype) * (1e-8 - min_eig)

        q_forecasts, q_realized_sites = self._simulate_clean_site_forecasts(
            rng=rng,
            nSamples=nSamples,
            nSteps=nSteps,
            dt=dt,
            horizon=horizon,
            n_sites=n_sites,
            target=target,
            kappa=kappa,
            sigma=sigma,
            corr_matrix=corr,
        )
        de = slice(0, sites_per_country)
        nl = slice(sites_per_country, n_sites)
        de_weights = np.full(sites_per_country, 1.0 / sites_per_country, dtype=self.np_dtype)
        nl_weights = np.full(sites_per_country, 1.0 / sites_per_country, dtype=self.np_dtype)
        q_de_forecast = np.sum(de_weights * q_forecasts[:, :, de], axis=2)
        q_nl_forecast = np.sum(nl_weights * q_forecasts[:, :, nl], axis=2)
        q_de_total = np.sum(de_weights * q_realized_sites[:, de], axis=1)
        q_nl_total = np.sum(nl_weights * q_realized_sites[:, nl], axis=1)
        q_de_cong = np.sum(de_weights[:split] * q_realized_sites[:, :split], axis=1)
        q_de_unc = np.sum(de_weights[split:] * q_realized_sites[:, split:sites_per_country], axis=1)
        q_de_tilde = np.minimum(q_de_cong, l_max_de) + q_de_unc
        q_disp_de = np.sqrt(np.mean((q_forecasts[:, :, de] - q_de_forecast[:, :, np.newaxis]) ** 2, axis=2))
        q_disp_nl = np.sqrt(np.mean((q_forecasts[:, :, nl] - q_nl_forecast[:, :, np.newaxis]) ** 2, axis=2))

        z_price = rng.normal(size=(nSamples, nSteps))
        z_price_nl = rho_price * z_price + np.sqrt(1.0 - rho_price**2) * rng.normal(size=(nSamples, nSteps))
        xp_de = np.zeros((nSamples, nSteps + 1), dtype=self.np_dtype)
        xp_nl = np.zeros((nSamples, nSteps + 1), dtype=self.np_dtype)
        persistence_p = np.exp(-kappa_p * dt)
        step_p = sigma_p * np.sqrt((1.0 - np.exp(-2.0 * kappa_p * dt)) / (2.0 * kappa_p))
        for t in range(1, nSteps + 1):
            xp_de[:, t] = persistence_p * xp_de[:, t - 1] + step_p * z_price[:, t - 1]
            xp_nl[:, t] = persistence_p * xp_nl[:, t - 1] + step_p * z_price_nl[:, t - 1]

        f_de = np.zeros((nSamples, nSteps + 1), dtype=self.np_dtype)
        f_nl = np.zeros((nSamples, nSteps + 1), dtype=self.np_dtype)
        g0_de = 1.0 - float(np.mean(q_de_forecast[:, 0]))
        g0_nl = 1.0 - float(np.mean(q_nl_forecast[:, 0]))
        for t in range(nSteps + 1):
            time_to_delivery = horizon - (t * dt)
            mt_de = xp_de[:, t] * np.exp(-kappa_p * time_to_delivery)
            mt_nl = xp_nl[:, t] * np.exp(-kappa_p * time_to_delivery)
            f_de[:, t] = f_0_T_de * (1.0 + mt_de) * ((1.0 - q_de_forecast[:, t]) / g0_de)
            f_nl[:, t] = f_0_T_nl * (1.0 + mt_nl) * ((1.0 - q_nl_forecast[:, t]) / g0_nl)

        payoff = q_de_tilde * (f_de[:, -1] - ppa_strike_de)
        dInsts = np.stack(
            [f_de[:, -1, np.newaxis] - f_de[:, :nSteps], f_nl[:, -1, np.newaxis] - f_nl[:, :nSteps]],
            axis=-1,
        )
        price = np.stack([f_de[:, :nSteps], f_nl[:, :nSteps]], axis=-1)
        cost, ubnd_a, lbnd_a, ubnd_trade, lbnd_trade = self._clean_bounds_and_costs(
            nSamples=nSamples,
            nSteps=nSteps,
            nInst=2,
            price=price,
            ubnd_a_value=ubnd_a_value,
            lbnd_a_value=lbnd_a_value,
            ubnd_trade_value=ubnd_trade_value,
            lbnd_trade_value=lbnd_trade_value,
            use_transaction_cost=use_transaction_cost,
        )

        per_step_features = pdct(
            time_left=np.full(
                (nSamples, nSteps),
                np.linspace(float(nSteps), 1.0, nSteps, endpoint=True, dtype=self.np_dtype)[np.newaxis, :] * dt,
                dtype=self.np_dtype,
            ),
            forward_price=price,
            cost=cost,
        )
        if feature_model_type == "A":
            per_step_features.wind_info = np.stack(
                [q_de_forecast[:, :-1], q_nl_forecast[:, :-1]], axis=-1
            )
        elif feature_model_type == "B":
            per_step_features.wind_info = np.stack(
                [
                    q_de_forecast[:, :-1],
                    q_nl_forecast[:, :-1],
                    q_disp_de[:, :-1],
                    q_disp_nl[:, :-1],
                ],
                axis=-1,
            )
        elif feature_model_type == "C":
            per_step_features.wind_info = q_forecasts[:, :-1, :]
        else:
            raise ValueError("feature_model_type must be one of {'A', 'B', 'C'}.")

        curtailment = q_de_total - q_de_tilde
        details = pdct(
            forward_price=np.stack([f_de, f_nl], axis=-1),
            forward_price_de=f_de,
            forward_price_nl=f_nl,
            q_de_forecast=q_de_forecast,
            q_nl_forecast=q_nl_forecast,
            q_total_realized=q_de_total + q_nl_total,
            q_de_realized=q_de_total,
            q_nl_realized=q_nl_total,
            q_cong=q_de_cong,
            q_unc=q_de_unc,
            q_cong_flow=q_de_cong,
            q_tilde=q_de_tilde,
            q_curtailment=curtailment,
            q_congestion_flow=q_de_cong,
            q_congestion_region_realized=q_de_cong,
            q_congestion_curtailment_ratio=np.divide(
                q_de_cong - np.minimum(q_de_cong, l_max_de),
                q_de_cong,
                out=np.zeros_like(q_de_cong),
                where=q_de_cong > 1e-12,
            ),
            payoff=payoff,
            path_history=np.stack([f_de, f_nl, q_de_forecast, q_nl_forecast], axis=-1),
            cannibal_rat=np.ones(n_sites, dtype=self.np_dtype),
            site_forecasts=q_forecasts,
            site_realized=q_realized_sites,
            congestion_region_indices=np.arange(split, dtype=int),
            q_cong_ratio_realized=np.divide(
                q_de_cong,
                q_de_total,
                out=np.zeros_like(q_de_cong),
                where=q_de_total > 1e-12,
            ),
            l_max=np.array(l_max_de, dtype=self.np_dtype),
        )
        site_country_codes = np.array([0] * sites_per_country + [1] * sites_per_country, dtype=int)
        site_type_codes = np.array([1] * split + [0] * (sites_per_country - split) + [0] * sites_per_country, dtype=int)
        self._store_clean_world(
            config=config,
            nSamples=nSamples,
            nSteps=nSteps,
            dt=dt,
            payoff=payoff,
            dInsts=dInsts,
            price=price,
            cost=cost,
            ubnd_a=ubnd_a,
            lbnd_a=lbnd_a,
            ubnd_trade=ubnd_trade,
            lbnd_trade=lbnd_trade,
            per_step_features=per_step_features,
            details=details,
            latent_model_type="simple_cross_border_extension",
            feature_model_type=feature_model_type,
            ppa_contract="de_regional_volume_risk",
            use_transaction_cost=use_transaction_cost,
            inst_names=["DE Forward", "NL Forward"],
            site_country_codes=site_country_codes,
            site_type_codes=site_type_codes,
            site_country_names=["DE", "NL"],
            site_type_names=["unconstrained", "congested"],
        )

    def __init__(self, config: Config, dtype=dh_dtype):
        self.tf_dtype = dtype
        self.np_dtype = dtype.as_numpy_dtype()
        self.unique_id = None  # for serialization; see below
        self.config = config.copy()  # for cloning

        requested_world = None
        config_input = {}
        try:
            config_input = config.input_dict()
            requested_world = config_input.get("latent_model_type", None)
        except Exception:
            requested_world = None

        requested_legacy_model = config_input.get("model_type", None)
        if requested_world is None and isinstance(requested_legacy_model, str):
            if requested_legacy_model.strip().lower() == "replication":
                requested_world = "replication"

        if isinstance(requested_world, str):
            requested_world = requested_world.strip().lower()

        if requested_world in {
            "replication",
            "biegler_koenig_replication",
            "bk_replication",
            "clean_replication",
        }:
            self._init_biegler_koenig_replication_world(config)
            return

        if requested_world in {
            "spatial_volume_risk",
            "ten_site_spatial_volume_risk",
            "bk_10_site_spatial_volume_risk",
            "bk_10_site_spatial_volume_risk_basis",
        }:
            self._init_spatial_volume_risk_world(config)
            return

        if requested_world in {
            "simple_cross_border_extension",
            "cross_border_extension",
            "spatial_volume_risk_cross_border_extension",
            "simple_spatial_volume_risk_cross_border_extension",
        }:
            self._init_simple_cross_border_extension_world(config)
            return

        raise ValueError(
            "Unknown PPAWorld latent_model_type. Use one of "
            "{'biegler_koenig_replication', 'spatial_volume_risk', "
            "'bk_10_site_spatial_volume_risk_basis', "
            "'simple_cross_border_extension'}. "
            "The old monolithic PPAWorld implementation has been archived."
        )


    def get_static_pnl(self, delta):
        """Berekent de P&L vector voor een specifieke vaste delta."""
        ppa_payoff = self.details.payoff
        price_paths = np.asarray(self.details.forward_price)
        if price_paths.ndim == 2:
            price_change = price_paths[:, -1] - price_paths[:, 0]
            deltas = np.asarray(delta, dtype=self.np_dtype).reshape(())
            hedge_pnl = deltas * (-price_change)
            entry_cost = np.abs(deltas) * self.data.market.cost[:, 0, 0]
        else:
            price_change = price_paths[:, -1, :] - price_paths[:, 0, :]
            deltas = np.asarray(delta, dtype=self.np_dtype).reshape((1, self.nInst))
            hedge_pnl = np.sum(deltas * (-price_change), axis=1)
            entry_cost = np.sum(
                np.abs(deltas) * self.data.market.cost[:, 0, :], axis=1
            )
        if self.use_transaction_cost:
            return ppa_payoff + hedge_pnl - entry_cost
        return ppa_payoff + hedge_pnl

    def get_optimal_static_delta(self, lmbda=19.0):
        """
        Vindt de delta die de 5% ES (CVaR) minimaliseert.
        lmbda=19.0 komt overeen met 5% ES in jouw objectives.py.
        """
        from .objectives import oce_utility

        def optimize_static_delta(objective, n_inst):
            static_upper = float(np.nanmax(np.asarray(self.data.market.ubnd_a)))
            static_upper = max(static_upper, 1.0)
            bounds = [(0.0, static_upper)] * n_inst
            candidate_levels = np.linspace(0.0, static_upper, 5)
            candidates = np.array(
                np.meshgrid(*([candidate_levels] * n_inst)), dtype=self.np_dtype
            ).T.reshape(-1, n_inst)

            scored = [(float(objective(candidate)), candidate) for candidate in candidates]
            scored.sort(key=lambda item: item[0])
            best_value, best_delta = scored[0]

            for _, start_delta in scored[: min(5, len(scored))]:
                res = minimize(
                    objective,
                    x0=start_delta,
                    bounds=bounds,
                    method="Powell",
                    options={"maxiter": 200, "xtol": 1e-5, "ftol": 1e-6},
                )
                if res.success and float(res.fun) < best_value:
                    best_value = float(res.fun)
                    best_delta = np.asarray(res.x, dtype=self.np_dtype)

            return np.asarray(best_delta, dtype=self.np_dtype).reshape(n_inst)

        # Haal data op
        payoffs = self.details.payoff
        price_paths = np.asarray(self.details.forward_price)
        if price_paths.ndim == 2:
            price_changes = price_paths[:, -1] - price_paths[:, 0]
            cost_at_entry = self.data.market.cost[:, 0, 0]

            def objective(d):
                pnl = payoffs + d * (-price_changes)
                if self.use_transaction_cost:
                    pnl = pnl - np.abs(d) * cost_at_entry
                return -oce_utility(utility="cvar", lmbda=lmbda, X=pnl)

            optimal_delta = float(
                optimize_static_delta(lambda x: objective(float(x[0])), 1)[0]
            )
            _log.info(f"Optimale Statische Delta gevonden: {optimal_delta:.4f}")
            return optimal_delta

        price_changes = price_paths[:, -1, :] - price_paths[:, 0, :]
        cost_at_entry = self.data.market.cost[:, 0, :]

        def objective(d):
            deltas = np.asarray(d, dtype=self.np_dtype).reshape((1, self.nInst))
            pnl = payoffs + np.sum(deltas * (-price_changes), axis=1)
            if self.use_transaction_cost:
                pnl = pnl - np.sum(np.abs(deltas) * cost_at_entry, axis=1)
            return -oce_utility(utility="cvar", lmbda=lmbda, X=pnl)

        optimal_delta = optimize_static_delta(objective, self.nInst)
        _log.info("Optimale Statische Delta gevonden: %s", optimal_delta)
        return optimal_delta

    def clone(self, config_overwrite=Config(), **kwargs):
        """
        Create a copy of this world with the same config, except for the seed.
        Used to generate genuine validation sets.

        Parameters
        ----------
            config_overwrite : Config, optional
                Allows specifying additional overwrites of specific config values
            **kwargs
                Allows specifying additional overwrites of specific config values, e.g.
                    world.clone( seed=222, samples=10 )
                If seed is not specified, a random seed is generated.

        Returns
        -------
            New world
        """
        if "seed" not in kwargs:
            kwargs["seed"] = int(np.random.randint(0, 0x7FFFFFFF))
        config = self.config.copy()
        config.update(config_overwrite, **kwargs)
        return PPAWorld(config)

    def plot(self, config=Config(), **kwargs):
        """Plot simple world"""

        config.update(kwargs)
        col_size = config.fig("col_size", 5, int, "Figure column size")
        row_size = config.fig("row_size", 5, int, "Figure row size")
        plot_samples = config("plot_samples", 5, int, "Number of samples to plot")
        print_input = config(
            "print_input",
            True,
            bool,
            "Whether to print the config inputs for the world",
        )

        xSamples = np.linspace(
            0, self.nSamples, plot_samples, endpoint=False, dtype=int
        )
        timeline1 = (
            np.cumsum(
                np.linspace(
                    0.0, self.nSteps, self.nSteps + 1, endpoint=True, dtype=np.float32
                )
            )
            * self.dt
        )
        timeline = timeline1[:-1]

        print(self.config.usage_report())

        fig = figure(tight=True, col_size=col_size, row_size=row_size, col_nums=3)
        fig.suptitle(self.__class__.__name__, fontsize=16)

        # forward price
        ax = fig.add_plot()
        ax.set_title("Forward Price")
        ax.set_xlabel("Time")
        if self.details.forward_price.ndim == 3:
            for i in xSamples:
                ax.plot(timeline1, self.details.forward_price[i, :, 0], "-", color="tab:blue", alpha=0.35)
                ax.plot(timeline1, self.details.forward_price[i, :, 1], "-", color="tab:orange", alpha=0.35)
            ax.plot(
                timeline1,
                np.mean(self.details.forward_price[:, :, 0], axis=0),
                "-",
                color="tab:blue",
                label="NL mean",
            )
            ax.plot(
                timeline1,
                np.mean(self.details.forward_price[:, :, 1], axis=0),
                "-",
                color="tab:orange",
                label="DE mean",
            )
        else:
            for i, color in zip(xSamples, colors_tableau()):
                ax.plot(timeline1, self.details.forward_price[i, :], "-", color=color)
            ax.plot(
                timeline1,
                np.mean(self.details.forward_price, axis=0),
                "_",
                color="black",
                label="mean",
            )
        ax.legend()

        ax = fig.add_plot()
        if hasattr(self.details, "q_nl_forecast"):
            ax.set_title("Country Wind Forecast")
            ax.set_ylabel("time")
            for i in xSamples:
                ax.plot(timeline1, self.details.q_nl_forecast[i, :], "-", color="tab:blue", alpha=0.35)
                ax.plot(timeline1, self.details.q_de_forecast[i, :], "-", color="tab:orange", alpha=0.35)
            ax.plot(
                timeline1,
                np.mean(self.details.q_nl_forecast, axis=0),
                "-",
                color="tab:blue",
                label="NL forecast",
            )
            ax.plot(
                timeline1,
                np.mean(self.details.q_de_forecast, axis=0),
                "-",
                color="tab:orange",
                label="DE forecast",
            )
        else:
            ax.set_title("Aggregate Wind Forecast")
            ax.set_ylabel("time")
            for i, color in zip(xSamples, colors_tableau()):
                ax.plot(timeline1, self.details.q_agg_forecast[i, :], "-", color=color)
            ax.plot(
                timeline1,
                np.mean(self.details.q_agg_forecast, axis=0),
                "_",
                color="black",
                label="mean",
            )
        ax.legend()

        ax = fig.add_plot()
        if hasattr(self.details, "q_nl_delivered"):
            ax.set_title("Delivered Volume")
            ax.set_ylabel("time")
            ax.plot(
                timeline1,
                np.full_like(timeline1, np.mean(self.details.q_nl_delivered)),
                "-",
                color="tab:blue",
                label="NL delivered mean",
            )
            ax.plot(
                timeline1,
                np.full_like(timeline1, np.mean(self.details.q_de_delivered)),
                "-",
                color="tab:orange",
                label="DE delivered mean",
            )
        else:
            ax.set_title("Congested vs Uncongested")
            ax.set_ylabel("volume")
            ax.plot(
                timeline1,
                np.full_like(timeline1, np.mean(self.details.q_cong)),
                "-",
                color="tab:red",
                label="q_cong mean",
            )
            ax.plot(
                timeline1,
                np.full_like(timeline1, np.mean(self.details.q_tilde)),
                "-",
                color="tab:green",
                label="q_tilde mean",
            )
        ax.legend()

        fig.render()
        del fig

        if print_input:
            print("Config settings:\n%s" % self.input_report)

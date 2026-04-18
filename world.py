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
from scipy.optimize import bisect

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
    def __init__(self, config: Config, dtype=dh_dtype):
        self.tf_dtype = dtype
        self.np_dtype = dtype.as_numpy_dtype()
        self.unique_id = None  # for serialization; see below
        self.config = config.copy()  # for cloning

        # simulator
        # ---------
        # nSteps     = config("steps", 10, int, help="Number of time steps")
        # nSamples   = config("samples", 1000, int, help="Number of samples")
        # seed       = config("seed", 2312414312, int, help="Random seed")
        # nIvSteps   = config("invar_steps", 5, int, help="Number of steps ahead to sample from invariant distribution")
        # dt         = config("dt", 1./50., float, help="Time per timestep.", help_default="One week (1/50)")
        # cost_s     = config("cost_s", 0.0002, float, help="Trading cost spot")
        # ubnd_as    = config("ubnd_as", 5., float, help="Upper bound for the number of shares traded at each time step")
        # lbnd_as    = config("lbnd_as", -5., float, help="Lower bound for the number of shares traded at each time step")
        # bs_mode    = config("black_scholes", False, bool, help="Hard overwrite to use a black & scholes model with vol 'rvol' and drift 'drift'. Also turns off the option as a tradable instrument by setting strike = 0.")
        # no_svol    = config("no_stoch_vol", False, bool, help="If true, turns off stochastic realized and implied vol, by setting meanrev_*vol = 0 and volvol_*vol = 0")
        # no_sdrift  = config("no_stoch_drift", False, bool, help="If true, turns off the stochastic drift of the asset, by setting meanrev_drift = 0. and drift_vol = 0")
        # _log.verify( nSteps > 0,    "'steps' must be positive; found %ld", nSteps )
        # _log.verify( nSamples > 0,  "'samples' must be positive; found %ld", nSamples )
        # _log.verify( dt > 0., "dt must be positive; found %g", dt )
        # _log.verify( cost_s >= 0, "'cost_s' must not be negative; found %g", cost_s )
        # _log.verify( ubnd_as >= 0., "'ubnd_as' must not be negative; found %g", ubnd_as )
        # _log.verify( lbnd_as <= 0., "'lbnd_as' must not be positive; found %g", lbnd_as )
        # _log.verify( ubnd_as - lbnd_as > 0., "'ubnd_as - lbnd_as' must be positive; found %g", ubnd_as - lbnd_as)

        # sqrt_time_left = np.sqrt( time_left )

        def ou_stats(x0, kappa, sigma, t, T):
            mean = x0 * np.exp(-kappa * (T - t))
            var = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * (T - t)))
            return mean, var

        def ou_call_price(K, T, kappa, sigma, x0=0, mu=0):
            """
            Docstring for ou_call_price

            :param K: Description
            :param T: Description
            :param kappa: Description
            :param sigma: Description
            :param x0: Description
            :param mu: Description
            """
            mean_T = x0 * np.exp(-kappa * T) + mu * (1 - np.exp(-kappa * T))

            variance_T = (sigma**2 / (2 * kappa)) * (1 - np.exp(-2 * kappa * T))
            std_T = np.sqrt(variance_T)

            d = (mean_T - K) / std_T

            # Final Price Formula
            price = (mean_T - K) * norm.cdf(d) + std_T * norm.pdf(d)

            return price

        def gen_sigmoid_weights(strikes):
            """
            Generates weights w_j such that sum of calls approximates sigmoid/

            :param strikes: Grid of NxN containing the strikes of the call options
            """
            varsigma = lambda x: 1.0 / (1.0 + np.exp(-x))
            vals = varsigma(strikes)
            w = np.zeros(len(strikes))
            # weights for linear interpolation on the grid
            w[0] = (vals[1] - vals[0]) / (strikes[1] - strikes[0])
            for j in range(1, len(strikes) - 1):
                w[j] = (vals[j + 1] - vals[j]) / (strikes[j + 1] - strikes[j]) - (
                    vals[j] - vals[j - 1]
                ) / (strikes[j] - strikes[j - 1])
            return w

        def ppa_payoff(path_history):
            """
            path_history: tensor/array van vorm (nSamples, nSteps + 1, nFeatures)
            Feature 0: Forward Price f(t,T)
            Feature 1: Onshore Infeed realization/forecast
            """
            f_T_T = path_history[:, -1, 0]  # Spot price at T
            q_T = path_history[:, -1, 1]  # Realized infeed at T

            strike = 100.0
            capacity = 1.0

            # Payoff voor de offtaker (buyer of the PPA)
            payoff = capacity * q_T * (f_T_T - strike)

            return payoff

        def get_expected_q(x_t, T_rem, kappa, sigma, phi):
            """
            Berekent E[sigmoid(X_T + phi) | F_t]
            door de sigmoid te benaderen als gewogen som van call opties.
            """
            if T_rem <= 0:
                # At final date, the expectation is equal to the reality
                return 1.0 / (1.0 + np.exp(-(x_t + phi)))

            res = np.zeros_like(x_t)
            for j in range(len(strikes)):
                # Use the analytical price of a call of an OU process
                res += sigmoid_weights[j] * ou_call_price(
                    K=strikes[j] - phi, T=T_rem, kappa=kappa, sigma=sigma, x0=x_t
                )
            return res

        def get_expected_q_vectored(x_t, T_rem, kappa, sigma, phi):
            if T_rem <= 0:
                return 1.0 / (1.0 + np.exp(-(x_t + phi)))

            s = strikes[:, np.newaxis]
            w = sigmoid_weights[:, np.newaxis]

            prices = ou_call_price(K=s - phi, T=T_rem, kappa=kappa, sigma=sigma, x0=x_t)

            return np.sum(w * prices, axis=0)

        def calibrate_phi(target_forecast, kappa, sigma, T_max):
            """
            Zoekt de waarde van phi waarvoor de verwachting op t=0
            gelijk is aan de marktvoorspelling (bijv. 0.5).
            """

            def objective(phi_guess):
                # At t=0 is x0 = 0
                val = get_expected_q(np.array([0.0]), T_max, kappa, sigma, phi_guess)
                return val[0] - target_forecast

            # Use bisect to find the roots
            return bisect(objective, -5, 5)

        def varsigma(x):
            return 1.0 / (1.0 + np.exp(-x))

        strikes = np.linspace(-5, 5, 20)
        sigmoid_weights = gen_sigmoid_weights(strikes)

        nSamples = config("samples", 100000, int, help="Number of simulated paths")
        nSteps = config("steps", 48, int, help="Number of time steps")
        T_max = config("max_time", 48, int, help="Maximum time")
        dt = config("dt_replicate", T_max / nSteps, float, help="Time per timestep")
        time_left = (
            np.linspace(float(nSteps), 1.0, nSteps, endpoint=True, dtype=self.np_dtype)
            * dt
        )

        # OU parameters
        kappa_q2 = config(
            "mean_rev_q2", 0.1 / 24.0, float, help="Mean reversion offshore"
        )
        sigma_q2 = config(
            "vol_q2", 3.0 / np.sqrt(365.0 * 24.0), float, help="Volatility offshore"
        )
        kappa_q1 = config(
            "mean_rev_q1", 0.1 / 24.0, float, help="Mean reversion offshore"
        )
        sigma_q1 = config(
            "vol_q1", 3.0 / np.sqrt(365.0 * 24.0), float, help="Volatility offshore"
        )

        kappa_p = config("mean_rev_p", 0.5 / 24.0, float, help="General mean reversion")
        sigma_p = config(
            "vol_p", 0.8 / np.sqrt(365.0 * 24.0), float, help="General std dev"
        )
        rho = config(
            "rho", 0.46, float, help="Correlation between onshore and offshore"
        )

        kappa_agg = config(
            "mean_rev_agg",
            DEFAULT_A_AGG,
            float,
            help="Mean reversion of the aggregate wind capacity of the 15 locations.",
        )
        sigma_agg = config(
            "sigma_agg",
            DEFAULT_SIGMA_AGG,
            float,
            help="Aggregated diffusion of the 15 locations",
        )

        kappa_full = config(
            "mean_reversion_mat",
            DEFAULT_A_MATRIX.tolist(),
            list,
            help="Mean reversion matrix",
        )
        sigma_full = config(
            "vol_spatial_field",
            DEFAULT_SIGMA_MATRIX.tolist(),
            list,
            help="Full spatial field of autocorrelated shocks",
        )

        q_target = config(
            "q_target", 0.5, float, help="target for wind aggregate at time 0"
        )

        # forward information at t=0
        f_0_T = config("f_0_T", 100.0, float, help="Forward price at time 0")
        q1_target = config(
            "q1_target", 0.5, float, help="target for wind onshore at time 0"
        )
        q2_target = config("q2_target", 0.5, float, help="target for wind offshore")
        w1 = config("w1", 0.8, float, help="weight of onshore renewable infeed")
        w2 = config("w2", 0.2, float, help="weight of offshore renewable infeed")

        model_type = config(
            "model_type",
            "A",
            str,
            help="Model choices. {A, B, C} => {aggregate, agg + dispersion, full spatial field}",
        )

        if model_type == "Replication":
            num_dim = 2
            k_vals = np.array([kappa_q1, kappa_q2])
            s_vals = np.array([sigma_q1, sigma_q2])
            targets = np.array([q1_target, q2_target])
            model_weights = np.array([w1, w2])

            persistence = np.diag(np.exp(-k_vals * dt))
            cov_matrix = np.array([
                [sigma_q1**2, rho * sigma_q1 * sigma_q2],
                [rho * sigma_q1*sigma_q2, sigma_q2**2]
            ])
            discrete_cov = cov_matrix * ((1 - np.exp(-2 * kappa_q1 * dt)) / (2 * kappa_q1))
            vol_step = np.linalg.cholesky(discrete_cov)

            # persistence = np.exp(-k_vals * dt)
            # vol_step = s_vals * np.sqrt((1 - np.exp(-2 * k_vals * dt)) / (2 * k_vals))
        elif model_type == "synthethic_field":
            num_dim = 10
            length_scale = config("length_scale", 50.0, float, help="Spatial correlation length")
            coords = np.array([[i * 10.0, 0.0] for i in range(num_dim)])
            dist_matrix = cdist(coords, coords, metric='euclidean')

            cov_matrix = (sigma_q1**2) * np.exp(-dist_matrix / length_scale)

            k_vals = np.full(num_dim, kappa_q1)
            s_vals = np.full(num_dim, sigma_q1)
            targets = np.full(num_dim, q_target)
            model_weights = np.ones(num_dim) / num_dim
            # 4. Matrices for the simulation loop
            persistence = np.diag(np.exp(-k_vals * dt)) 
            
            # Exact discrete step variance matrix
            discrete_cov = cov_matrix * ((1 - np.exp(-2 * kappa_q1 * dt)) / (2 * kappa_q1))
            vol_step = np.linalg.cholesky(discrete_cov)
        
        else:
            num_dim = 15
            A_mat = np.array(kappa_full)
            Sigma_mat = np.array(sigma_full)
            k_vals = np.array(np.diag(A_mat) / dt)
            s_vals = np.diag(Sigma_mat) / np.sqrt(dt)
            targets = np.full(15, q_target)
            model_weights = np.ones(15) / 15

            persistence = A_mat
            vol_step = Sigma_mat
        # elif model_type == "A":
        #     k_vals = np.array([-np.log(kappa_agg) / dt])
        #     s_vals = np.array([sigma_agg / np.sqrt(dt)])
        #     targets = np.array([q_target])
        #     model_weights = np.array([1.0])
        #     persistence = np.array([kappa_agg])
        #     vol_step = np.array([sigma_agg])
        # elif model_type == "B":
        #     k_vals = np.array([-np.log(kappa_agg) / dt])
        #     s_va

        # Times when the weather forecasts are becoming "known" to the power prices -> not in numerical experimets only for figure!
        update_times = [0, 10, 14, 18, 34, 38, 42]

        # calibration of phi and g0
        # print(sigma_full)
        # print(sigma_full.shape)
        phis = np.array(
            [
                calibrate_phi(targets[i], k_vals[i], s_vals[i], T_max)
                for i in range(len(targets))
            ]
        )

        E_q_0 = np.array(
            [
                get_expected_q_vectored(
                    np.array([0.0]), T_max, k_vals[i], s_vals[i], phis[i]
                )[0]
                for i in range(len(targets))
            ]
        )

        g0 = 1.0 - np.sum(model_weights * E_q_0)

        num_dim = len(targets)
        x = np.zeros((nSamples, nSteps + 1, num_dim))
        xp = np.zeros((nSamples, nSteps + 1))
        f_t_T = np.zeros((nSamples, nSteps + 1))
        q_forecasts = np.zeros((nSamples, nSteps + 1, num_dim))

        # # calibration of phi's to ensure unbiased start values
        # phi_q1 = calibrate_phi(q1_target, kappa_q1, sigma_q1, T_max)
        # phi_q2 = calibrate_phi(q2_target, kappa_q2, sigma_q2, T_max)

        # phi_agg = calibrate_phi(q_target, continuous_kappa, continuous_sigma, T_max )

        # # calibration of g0
        # E_q_0 = get_expected_q(np.array([0.0]), T_max, kappa_agg, sigma_agg, phi_agg)[0]
        # g0 = 1.0 - E_q_0
        # # replication of article
        # # E_q1_0 = get_expected_q(np.array([0.0]), T_max, kappa_q1, sigma_q1, phi_q1)[0]
        # # E_q2_0 = get_expected_q(np.array([0.0]), T_max, kappa_q2, sigma_q2, phi_q2)[0]
        # # g0 = 1.0 - (w1 * E_q1_0 + w2 * E_q2_0)

        # # idiosyncratic start value ??
        # m0_mean, m0_var = ou_stats(0.0, kappa_p, sigma_p, 0, T_max)
        # idio0 = 1.0

        # # Simulatie initialisatie
        # x1 = np.zeros((nSamples, nSteps + 1))
        # x2 = np.zeros((nSamples, nSteps + 1))
        # xp = np.zeros((nSamples, nSteps + 1))
        # f_t_T = np.zeros((nSamples, nSteps + 1))
        # q1_t_T = np.zeros((nSamples, nSteps + 1))
        # q2_t_T = np.zeros((nSamples, nSteps + 1))

        # x_agg = np.zeros((nSamples, nSteps + 1))
        # xp = np.zeros((nSamples, nSteps + 1))
        # f_t_T = np.zeros((nSamples, nSteps + 1))
        # q_t_T = np.zeros((nSamples, nSteps + 1))

        for t in range(nSteps + 1):
            t_curr = t * dt

            if t_curr in update_times:
                t_minus_idx = t

            T_rem = T_max - (t_minus_idx * dt)

            if t > 0:
                # update latent wind state x

                z = np.random.normal(size=(nSamples, num_dim))
                # if model_type == "Replication":
                #     z[:, 1] = rho * z[:, 0] + np.sqrt(1 - rho**2) * np.random.normal(
                #         size=nSamples
                #     )
                x[:, t, :] = (x[:, t - 1, :] @ persistence) + (z @ vol_step.T)

                zp = np.random.normal(size=nSamples)
                xp[:, t] = (
                    xp[:, t - 1] * np.exp(-kappa_p * dt)
                    + sigma_p
                    * np.sqrt((1 - np.exp(-2 * kappa_p * dt)) / (2 * kappa_p))
                    * zp
                )

                # OLD LOOP!
                # z1 = np.random.normal(size=nSamples)
                # z_raw = np.random.normal(size=nSamples)
                # z2 = rho * z1 + np.sqrt(1 - rho**2) * z_raw
                # zp = np.random.normal(size=nSamples)

                # # evolution of OU processes
                # x1[:, t] = (
                #     x1[:, t - 1] * np.exp(-kappa_q1 * dt)
                #     + sigma_q1
                #     * np.sqrt((1 - np.exp(-2 * kappa_q1 * dt)) / (2 * kappa_q1))
                #     * z1
                # )
                # x2[:, t] = (
                #     x2[:, t - 1] * np.exp(-kappa_q2 * dt)
                #     + sigma_q2
                #     * np.sqrt((1 - np.exp(-2 * kappa_q2 * dt)) / (2 * kappa_q2))
                #     * z2
                # )
                # xp[:, t] = (
                #     xp[:, t - 1] * np.exp(-kappa_p * dt)
                #     + sigma_p
                #     * np.sqrt((1 - np.exp(-2 * kappa_p * dt)) / (2 * kappa_p))
                #     * zp
                # )

            eq_m = np.zeros((nSamples, num_dim))
            for i in range(num_dim):
                eq_m[:, i] = get_expected_q(
                    x[:, t_minus_idx, i], T_rem, k_vals[i], s_vals[i], phis[i]
                )
            q_forecasts[:, t, :] = eq_m

            gt = 1.0 - np.sum(model_weights * eq_m, axis=1)

            mt_mean, _ = ou_stats(xp[:, t], kappa_p, sigma_p, t_curr, T_max)
            f_t_T[:, t] = f_0_T * (1.0 + mt_mean) * (gt / g0)
            # OLD PART!
            # # infeed forecasts
            # # deleted the update times, this led to not all information used to calculate the forward price.
            # eq1_m = get_expected_q(
            #     x1[:, t_minus_idx], T_rem, kappa_q1, sigma_q1, phi_q1
            # )
            # eq2_m = get_expected_q(
            #     x2[:, t_minus_idx], T_rem, kappa_q2, sigma_q2, phi_q2
            # )

            # q1_t_T[:, t] = eq1_m
            # q2_t_T[:, t] = eq2_m

            # # forward price
            # gt = 1.0 - (w1 * eq1_m + w2 * eq2_m)

            # # idiosyncratic component
            # mt_mean, mt_var = ou_stats(xp[:, t], kappa_p, sigma_p, t_curr, T_max)
            # idiot = 1.0 + mt_mean

            # # final power price at time t
            # f_t_T[:, t] = f_0_T * (idiot / idio0) * (gt / g0)

        q_realized_sites = 1.0 / (1.0 + np.exp(-(x[:, -1, :] + phis)))
        q_total_realized = np.sum(model_weights * q_realized_sites, axis=1)

        path_history = np.zeros((nSamples, nSteps + 1, 2), dtype=self.np_dtype)
        path_history[:, :, 0] = f_t_T

        q_agg_forecast = np.sum(model_weights * q_forecasts, axis=2)
        path_history[:, :-1, 1] = q_agg_forecast[:, :-1]
        path_history[:, -1, 1] = q_total_realized

        # OLD
        # path_history = np.zeros((nSamples, nSteps, 2))
        # path_history[:, :, 0] = f_t_T[:, 1:]

        # q1_realized = 1.0 / (1.0 + np.exp(-(x1[:, -1] + phi_q1)))
        # q2_realized = 1.0 / (1.0 + np.exp(-(x2[:, -1] + phi_q2)))

        # path_history[:, :, 1] = q1_t_T[:, 1:]
        # path_history[:, -1, 1] = q1_realized  # actual wind at T
        payoff = ppa_payoff(path_history)

        # hedging instrument(s)
        strike = 100.0
        dS = (
            f_t_T[:, nSteps][:, np.newaxis] - f_t_T[:, :nSteps]
        )  # this is for buying forwards, below is selling.
        # f_t_T[:, :nSteps] - f_t_T[:,nSteps][:, np.newaxis]

        dInsts = np.zeros((nSamples, nSteps, 1))

        dInsts[:, :, 0] = dS  # represents the payoff of forward contracts

        # bid_ask_spread_free = 0.015 # 1.5% spread from f_t_T -> spread
        
        spread_base = 0.01
        spread_max = 0.03
        tau = time_left
        lamb = 5.0

        bid_ask_spread_free = spread_base + spread_max * np.exp(-lamb * tau) # exponential time dependent spread

        cost_mwh = 0.15 # cost per mwH

        cost_matrix = (f_t_T * (bid_ask_spread_free / 2)) + cost_mwh

        cost = cost_matrix[:,:,np.newaxis]
        price = f_t_T[:, :nSteps]
        ubnd_a = np.full((nSamples, nSteps, 1), 5)
        lbnd_a = np.full((nSamples, nSteps, 1), -5)

        # -----------------------------
        # unique_id
        # -----------------------------
        # Default handling for configs will ignore any function definitions, e.g. in this case 'payoff'.
        # we therefore manually generate a sufficient hash
        self.unique_id = uniqueHash(
            [config.input_dict(), payoff, self.tf_dtype.name], parse_functions=True
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
        per_step_features = pdct(
            time_left=np.full(
                (nSamples, nSteps), time_left[np.newaxis, :], dtype=self.np_dtype
            ),
            forward_price=f_t_T[:, :-1],
            cost=cost,
        )
        if model_type in ["Replication", "A"]:
            per_step_features.wind_info = q_agg_forecast[:, :-1, np.newaxis]

        elif model_type == "B":
            # agent sees aggregate mean + spatial dispersion
            dispersion = np.var(q_forecasts[:, :-1, :], axis=2)
            per_step_features.wind_info = np.stack(
                [q_agg_forecast[:, :-1], dispersion], axis=-1
            )

        elif model_type == "C" or model_type == "synthetic_field":
            # Agent sees all 15 individual site forecasts
            per_step_features.wind_info = q_forecasts[:, :-1, :]
        
        self.data.features = pdct(per_step=per_step_features, per_path=pdct())

        # OLD
        # self.data.features = pdct(
        #     per_step=pdct(
        #         # both spot and option, if present
        #         # cost   = cost,            # trading cost
        #         time_left=np.full(
        #             (nSamples, nSteps), time_left[np.newaxis, :], dtype=self.np_dtype
        #         ),
        #         # forward_price = f_t_T[:,1:],
        #         # q1_forecast = q1_t_T[:,1:],
        #         # q2_forecast = q2_t_T[:, 1:],
        #         forward_price=f_t_T[:, :-1],  # Exclude t=T (spot price)
        #         q1_forecast=q1_t_T[:, :-1],  # Exclude realized wind at T
        #         q2_forecast=q2_t_T[:, :-1],
        #     ),
        #     per_path=pdct(),
        # )

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
            forward_price=f_t_T,
            q_agg_forecast=q_agg_forecast,
            q_total_realized=q_total_realized,
            payoff=payoff,
            path_history=path_history,
        )
        # Keep details strictly numeric so assert_iter_not_is_nan can validate it.
        if num_dim > 1:
            self.details.site_forecasts = q_forecasts
            self.details.site_realized = q_realized_sites

        # check numerics
        assert_iter_not_is_nan(self.details, "details")

        # generating sample weights
        # the tf_sample_weights is passed to keras train and must be of size [nSamples,1]
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
        # Static and dynamic volume hedging?

        # Feature engineering

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
        for i, color in zip(xSamples, colors_tableau()):
            ax.plot(timeline1, self.details.forward_price[i, :], "-", color=color)
        ax.plot(
            timeline1,
            np.mean(self.details.forward_price, axis=0),
            "_",
            color="black",
            label="mean",
        )
        #        ax.get_xaxis().get_major_formatter().get_useOffset(False)
        ax.legend()

        # q1 -> onshore wind
        ax = fig.add_plot()
        ax.set_title("Onshore wind infeed")
        ax.set_ylabel("time")
        for i, color in zip(xSamples, colors_tableau()):
            ax.plot(timeline1, self.details.onshore_wind[i, :], "-", color=color)
        ax.plot(
            timeline1,
            np.mean(self.details.onshore_wind, axis=0),
            "_",
            color="black",
            label="mean",
        )
        ax.legend()

        # q2 -> offshore wind
        ax = fig.add_plot()
        ax.set_title("Offshore wind infeed")
        ax.set_ylabel("time")
        for i, color in zip(xSamples, colors_tableau()):
            ax.plot(timeline1, self.details.offshore_wind[i, :], "-", color=color)
        ax.plot(
            timeline1,
            np.mean(self.details.offshore_wind, axis=0),
            "_",
            color="black",
            label="mean",
        )
        ax.legend()

        # # drift
        # ax  = fig.add_plot()
        # ax.set_title("Drift")
        # ax.set_xlabel("Time")
        # for i, color in zip( xSamples, colors_tableau() ):
        #     ax.plot( timeline, self.details.drift[i,:], "-", color=color )
        # ax.plot( timeline, np.mean( self.details.drift, axis=0), "_", color="black", label="mean" )

        # # vols
        # ax  = fig.add_plot()
        # ax.set_title("Volatilities")
        # ax.set_xlabel("Time")
        # for i, color in zip( xSamples, colors_tableau() ):
        #     ax.plot( timeline, self.data.features.per_step.ivol[i,:], "-", color=color )
        #     ax.plot( timeline, self.details.rvol[i,:], ":", color=color )

        # if self.nInst > 1:
        #     # call prices
        #     ax  = fig.add_plot(True)
        #     ax.set_title("Call Prices")
        #     ax.set_xlabel("Time")
        #     for i, color in zip( xSamples, colors_tableau() ):
        #         ax.plot( timeline, self.data.features.per_step.call_price[i,:], "-", color=color )
        #     ax.plot( timeline, np.mean( self.data.features.per_step.call_price, axis=0), "_", color="black", label="mean" )
        #     ax.legend()

        #     # call delta
        #     ax  = fig.add_plot()
        #     ax.set_title("Call Deltas")
        #     ax.set_xlabel("Time")
        #     for i, color in zip( xSamples, colors_tableau() ):
        #         ax.plot( timeline, self.data.features.per_step.call_delta[i,:], "-", color=color )
        #     ax.plot( timeline, np.mean( self.data.features.per_step.call_delta, axis=0), "_", color="black", label="mean" )
        #     ax.legend()

        #     # call vega
        #     ax  = fig.add_plot()
        #     ax.set_title("Call Vegas")
        #     ax.set_xlabel("Time")
        #     for i, color in zip( xSamples, colors_tableau() ):
        #         ax.plot( timeline, self.data.features.per_step.call_vega[i,:], "-", color=color )
        #     ax.plot( timeline, np.mean( self.data.features.per_step.call_vega, axis=0), "_", color="black", label="mean" )
        #     ax.legend()

        fig.render()
        del fig

        if print_input:
            print("Config settings:\n%s" % self.input_report)

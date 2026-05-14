### Code graveyard for the legacy version of clas PPAWorld

        # simulator
        # ---------

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

        def ppa_payoff(path_history, Q_tilde):
            """
            path_history: tensor/array van vorm (nSamples, nSteps + 1, nFeatures)
            Feature 0: Forward Price f(t,T)
            Feature 1: Onshore Infeed realization/forecast
            Feature 2: Wind production with congestion
            """
            f_T_T = path_history[:, -1, 0]  # Spot price at T
            q_T = path_history[:, -1, 1]  # Realized infeed at T

            ppa_strike = 100.0
            capacity = 1.0

            # Payoff voor de offtaker (buyer of the PPA)
            payoff =  Q_tilde * (f_T_T - ppa_strike)
            # payoff = capacity * Q_tilde * (f_T_T - ppa_strike)

            return payoff

        def get_expected_q(x_t, T_rem, kappa, sigma, phi):
            """
            Berekent E[sigmoid(X_T + phi) | F_t]
            door de sigmoid te benaderen als gewogen som van call opties.
            """
            if T_rem <= 0:
                # At final date, the expectation is equal to the reality
                return 1.0 / (1.0 + np.exp(-(x_t + phi)))

            if use_exact_sigmoid_expectation:
                mean_T = x_t * np.exp(-kappa * T_rem)
                var_T = (sigma**2 / (2 * kappa)) * (
                    1 - np.exp(-2 * kappa * T_rem)
                )
                nodes = mean_T[:, np.newaxis] + phi + np.sqrt(2.0 * var_T) * gh_x
                return np.sum(gh_w * (1.0 / (1.0 + np.exp(-nodes))), axis=1) / np.sqrt(
                    np.pi
                )

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

            if use_exact_sigmoid_expectation:
                mean_T = x_t * np.exp(-kappa * T_rem)
                var_T = (sigma**2 / (2 * kappa)) * (
                    1 - np.exp(-2 * kappa * T_rem)
                )
                nodes = mean_T[:, np.newaxis] + phi + np.sqrt(2.0 * var_T) * gh_x
                return np.sum(gh_w * (1.0 / (1.0 + np.exp(-nodes))), axis=1) / np.sqrt(
                    np.pi
                )

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
        gh_x, gh_w = np.polynomial.hermite.hermgauss(40)

        nSamples = config("samples", 100000, int, help="Number of simulated paths")
        nSteps = config("steps", 48, int, help="Number of time steps")
        seed = config("seed", 2312414312, int, help="Random seed")
        T_max = config("max_time", 48, int, help="Maximum time")
        dt = config("dt_replicate", T_max / nSteps, float, help="Time per timestep")
        time_left = (
            np.linspace(float(nSteps), 1.0, nSteps, endpoint=True, dtype=self.np_dtype)
            * dt
        )

        np.random.seed(seed)

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
        synthetic_mean_reversion = config(
            "synthetic_mean_reversion",
            kappa_q1,
            float,
            help="Mean reversion used by the synthetic latent wind field.",
        )
        synthetic_vol = config(
            "synthetic_vol",
            sigma_q1,
            float,
            help="Per-site volatility scale used by the synthetic latent wind field.",
        )
        synthetic_target = config(
            "synthetic_target",
            q_target,
            float,
            help="Common target level used by the synthetic latent wind field.",
        )
        synthetic_target_onshore = config(
            "synthetic_target_onshore",
            synthetic_target,
            float,
            help="Target wind level for onshore synthetic sites.",
        )
        synthetic_target_offshore = config(
            "synthetic_target_offshore",
            synthetic_target,
            float,
            help="Target wind level for offshore synthetic sites.",
        )
        synthetic_mean_reversion_onshore = config(
            "synthetic_mean_reversion_onshore",
            synthetic_mean_reversion,
            float,
            help="Mean reversion used by onshore synthetic sites.",
        )
        synthetic_mean_reversion_offshore = config(
            "synthetic_mean_reversion_offshore",
            synthetic_mean_reversion,
            float,
            help="Mean reversion used by offshore synthetic sites.",
        )
        synthetic_vol_onshore = config(
            "synthetic_vol_onshore",
            synthetic_vol,
            float,
            help="Volatility scale used by onshore synthetic sites.",
        )
        synthetic_vol_offshore = config(
            "synthetic_vol_offshore",
            synthetic_vol,
            float,
            help="Volatility scale used by offshore synthetic sites.",
        )
        synthetic_num_sites = config(
            "synthetic_num_sites",
            10,
            int,
            help="Number of synthetic sites in the synthetic latent wind field.",
        )
        synthetic_geography = config(
            "synthetic_geography",
            "single_country",
            str,
            help="Synthetic site layout. One of {'single_country', 'cross_border'}.",
        )
        synthetic_spacing = config(
            "synthetic_spacing",
            10.0,
            float,
            help="Distance between adjacent synthetic sites.",
        )
        synthetic_length_scale = config(
            "synthetic_length_scale",
            config(
                "length_scale",
                20.0,
                float,
                help="Legacy alias for synthetic_length_scale.",
            ),
            float,
            help="Spatial correlation length used by the synthetic latent wind field.",
        )
        synthetic_congestion_weight_spread = config(
            "synthetic_congestion_weight_spread",
            0.4,
            float,
            help="Spread of site-specific transmission-loading weights inside the congested block.",
        )
        synthetic_congestion_weight_shape = config(
            "synthetic_congestion_weight_shape",
            "linear",
            str,
            help="Shape of congestion-flow weights. One of {'linear', 'exponential'}.",
        )
        synthetic_model_b_features = config(
            "synthetic_model_b_features",
            "regional",
            str,
            help="Model B wind features for single-country local-cluster worlds. One of {'regional', 'aggregate_dispersion'}.",
        )
        synthetic_wind_dynamics = config(
            "synthetic_wind_dynamics",
            "ou_spatial",
            str,
            help=(
                "Synthetic wind dynamics. One of {'ou_spatial', "
                "'minimal_volume_risk', 'two_region_block_ou', "
                "'two_region_factor_sites'}."
            ),
        )
        synthetic_two_region_within_corr = config(
            "synthetic_two_region_within_corr",
            0.93,
            float,
            help=(
                "Within-region latent correlation for synthetic_wind_dynamics="
                "'two_region_block_ou' or 'two_region_factor_sites'."
            ),
        )
        synthetic_two_region_cross_corr = config(
            "synthetic_two_region_cross_corr",
            -0.35,
            float,
            help=(
                "Cross-region latent correlation for synthetic_wind_dynamics="
                "'two_region_block_ou' or 'two_region_factor_sites'."
            ),
        )
        synthetic_site_noise_vol = config(
            "synthetic_site_noise_vol",
            0.35,
            float,
            help=(
                "Relative site-level noise scale for synthetic_wind_dynamics="
                "'two_region_factor_sites'."
            ),
        )
        synthetic_local_curtailment_mode = config(
            "synthetic_local_curtailment_mode",
            "regional_pro_rata",
            str,
            help=(
                "Delivered-volume rule for local-cluster congestion worlds. "
                "Use 'regional_pro_rata' to curtail contracted sites by the regional "
                "flow ratio, or 'contract_cap' for min(Q_ppa_cong, L_max) + Q_ppa_unc."
            ),
        )
        synthetic_minimal_congested_sites = config(
            "synthetic_minimal_congested_sites",
            max(1, synthetic_num_sites // 2),
            int,
            help="Number of congested sites in the minimal volume-risk synthetic world.",
        )
        synthetic_minimal_national_loading = config(
            "synthetic_minimal_national_loading",
            1.0,
            float,
            help="National common factor loading in the minimal volume-risk synthetic world.",
        )
        synthetic_minimal_composition_loading = config(
            "synthetic_minimal_composition_loading",
            1.6,
            float,
            help="Opposite-sign congested/unconstrained composition loading in the minimal volume-risk synthetic world.",
        )
        synthetic_minimal_idio_loading = config(
            "synthetic_minimal_idio_loading",
            0.45,
            float,
            help="Site-idiosyncratic loading in the minimal volume-risk synthetic world.",
        )
        synthetic_minimal_price_wind_beta = config(
            "synthetic_minimal_price_wind_beta",
            90.0,
            float,
            help="Additive price sensitivity to uncurtailed national wind in the minimal spatial volume-risk world.",
        )
        synthetic_minimal_price_shock_vol = config(
            "synthetic_minimal_price_shock_vol",
            18.0,
            float,
            help="Terminal demand/residual-load price shock volatility in the minimal spatial volume-risk world.",
        )
        synthetic_ppa_cluster_mode = config(
            "synthetic_ppa_cluster_mode",
            "legacy",
            str,
            help=(
                "PPA site selection for the one-country local-cluster setup. "
                "One of {'legacy', 'mixed_low_loading', 'mixed_high', "
                "'mixed_extreme', 'congestion_high', 'single_high', 'explicit'}."
            ),
        )
        ppa_cluster_indices_override = config(
            "ppa_cluster_indices",
            [],
            list,
            help="Optional explicit site indices for the one-country local PPA cluster.",
        )
        ppa_contract_weights_override = config(
            "ppa_contract_weights",
            [],
            list,
            help=(
                "Optional contract weights aligned with ppa_cluster_indices. "
                "When omitted, national site weights are used as before."
            ),
        )
        ppa_capacity_scale = config(
            "ppa_capacity_scale",
            1.0,
            float,
            help=(
                "Capacity multiplier applied to the contracted local PPA cluster. "
                "Use roughly 1 / sum(country_weights[cluster]) to make a small cluster economically material."
            ),
        )
        save_synthetic_diagnostics = config(
            "save_synthetic_diagnostics",
            False,
            bool,
            help="Whether to save synthetic correlation and Cholesky heatmaps during world construction.",
        )
        l_max = config(
            "l_max", 100.0, float, help="Legacy congestion limit in the single-country setup"
        )
        l_max_nl = config(
            "l_max_nl",
            l_max,
            float,
            help="Delivered-volume cap applied to NL production in the cross-border setup.",
        )
        l_max_de = config(
            "l_max_de",
            l_max,
            float,
            help="Delivered-volume cap applied to DE production in the cross-border setup.",
        )

        # forward information at t=0
        f_0_T = config("f_0_T", 100.0, float, help="Forward price at time 0")
        f_0_T_nl = config(
            "f_0_T_nl",
            f_0_T,
            float,
            help="Forward price at time 0 for the Netherlands in the cross-border setup.",
        )
        f_0_T_de = config(
            "f_0_T_de",
            f_0_T,
            float,
            help="Forward price at time 0 for Germany in the cross-border setup.",
        )
        ppa_strike_nl = config(
            "ppa_strike_nl", 100.0, float, help="PPA strike for the Netherlands."
        )
        ppa_strike_de = config(
            "ppa_strike_de", 100.0, float, help="PPA strike for Germany."
        )
        ppa_contract = config(
            "ppa_contract",
            "portfolio",
            str,
            help="PPA payoff mode. One of {'portfolio', 'local_cluster', 'local_cluster_congestion', 'nl_single', 'nl_single_capture', 'local_portfolio_capture', 'nl_spatial_simple'}.",
        )
        ppa_capture_discount_nl_onshore = config(
            "ppa_capture_discount_nl_onshore",
            0.0,
            float,
            help="Capture-price discount for NL onshore production in nl_single_capture mode.",
        )
        ppa_capture_discount_nl_offshore = config(
            "ppa_capture_discount_nl_offshore",
            5.0,
            float,
            help="Structural capture-price basis discount for NL offshore production in nl_single_capture mode.",
        )
        ppa_capture_beta_onshore_nl_forward = config(
            "ppa_capture_beta_onshore_nl_forward",
            0.95,
            float,
            help="NL-forward beta of NL onshore production in nl_single_capture mode.",
        )
        ppa_capture_beta_onshore_de_forward = config(
            "ppa_capture_beta_onshore_de_forward",
            0.05,
            float,
            help="DE-forward beta of NL onshore production in nl_single_capture mode.",
        )
        ppa_capture_beta_offshore_nl_forward = config(
            "ppa_capture_beta_offshore_nl_forward",
            0.45,
            float,
            help="NL-forward beta of NL offshore production in nl_single_capture mode.",
        )
        ppa_capture_beta_offshore_de_forward = config(
            "ppa_capture_beta_offshore_de_forward",
            0.55,
            float,
            help="DE-forward beta of NL offshore production in nl_single_capture mode.",
        )
        ppa_capture_cannibalization_nl_offshore = config(
            "ppa_capture_cannibalization_nl_offshore",
            40.0,
            float,
            help="State-dependent capture-price discount per unit of NL offshore production.",
        )
        q1_target = config(
            "q1_target", 0.5, float, help="target for wind onshore at time 0"
        )
        q2_target = config("q2_target", 0.5, float, help="target for wind offshore")
        w1 = config("w1", 0.8, float, help="weight of onshore renewable infeed")
        w2 = config("w2", 0.2, float, help="weight of offshore renewable infeed")
        synthetic_country_spacing = config(
            "synthetic_country_spacing",
            500.0,
            float,
            help="Distance between the NL and DE synthetic site clusters.",
        )
        synthetic_offshore_offset = config(
            "synthetic_offshore_offset",
            100.0,
            float,
            help="Offset between onshore and offshore synthetic clusters.",
        )
        synthetic_price_beta_nl = config(
            "synthetic_price_beta_nl",
            1.0,
            float,
            help="Sensitivity of NL forward prices to NL aggregate wind forecasts.",
        )
        synthetic_price_beta_de = config(
            "synthetic_price_beta_de",
            1.0,
            float,
            help="Sensitivity of DE forward prices to DE aggregate wind forecasts.",
        )
        synthetic_country_price_corr = config(
            "synthetic_country_price_corr",
            0.7,
            float,
            help="Correlation between NL and DE country-specific price shocks.",
        )
        synthetic_national_corr = config(
            "synthetic_national_corr",
            0.35,
            float,
            help="Common national correlation floor for the one-country synthetic world.",
        )
        synthetic_local_cluster_idio = config(
            "synthetic_local_cluster_idio",
            0.35,
            float,
            help="Idiosyncratic loading scale in the one-country local-cluster factor model.",
        )
        kappa_p_shared = config(
            "mean_rev_p_shared",
            kappa_p,
            float,
            help="Mean reversion of the shared cross-border power-price factor.",
        )
        sigma_p_shared = config(
            "vol_p_shared",
            sigma_p,
            float,
            help="Volatility of the shared cross-border power-price factor.",
        )
        kappa_p_country = config(
            "mean_rev_p_country",
            kappa_p,
            float,
            help="Mean reversion of the country-specific power-price spread factors.",
        )
        sigma_p_country = config(
            "vol_p_country",
            sigma_p * 0.6,
            float,
            help="Volatility of the country-specific power-price spread factors.",
        )

        legacy_model_type = config(
            "model_type",
            None,
            str,
            help="Legacy combined selector. Prefer latent_model_type and feature_model_type.",
        )
        latent_model_type = config(
            "latent_model_type",
            None,
            str,
            help="Latent wind model. One of {replication, synthetic, era5field}.",
        )
        feature_model_type = config(
            "feature_model_type",
            None,
            str,
            help="Agent information set. One of {A, B, C}.",
        )
        use_transaction_cost = config(
            "use_transaction_cost",
            False,
            bool,
            help="Whether to include transaction costs in both dynamic and static hedging.",
        )
        ubnd_a = config(
            "ubnd_a",
            5.0,
            float,
            help="Upper bound for cumulative hedge positions in PPAWorld.",
        )
        lbnd_a = config(
            "lbnd_a",
            -5.0,
            float,
            help="Lower bound for cumulative hedge positions in PPAWorld.",
        )
        _log.verify(ubnd_a >= 0.0, "'ubnd_a' must not be negative; found %g", ubnd_a)
        _log.verify(lbnd_a <= 0.0, "'lbnd_a' must not be positive; found %g", lbnd_a)
        _log.verify(
            ubnd_a - lbnd_a > 0.0,
            "'ubnd_a - lbnd_a' must be positive; found %g",
            ubnd_a - lbnd_a,
        )
        ubnd_trade = config(
            "ubnd_trade",
            1.0e6,
            float,
            help="Upper bound for per-step hedge trades in PPAWorld.",
        )
        lbnd_trade = config(
            "lbnd_trade",
            -1.0e6,
            float,
            help="Lower bound for per-step hedge trades in PPAWorld.",
        )
        _log.verify(
            ubnd_trade >= 0.0,
            "'ubnd_trade' must not be negative; found %g",
            ubnd_trade,
        )
        _log.verify(
            lbnd_trade <= 0.0,
            "'lbnd_trade' must not be positive; found %g",
            lbnd_trade,
        )
        _log.verify(
            ubnd_trade - lbnd_trade > 0.0,
            "'ubnd_trade - lbnd_trade' must be positive; found %g",
            ubnd_trade - lbnd_trade,
        )

        if isinstance(legacy_model_type, str):
            legacy_model_type = legacy_model_type.strip()
            if legacy_model_type.lower() in {"", "none"}:
                legacy_model_type = None
            elif legacy_model_type.lower() in {"a", "b", "c"}:
                legacy_model_type = legacy_model_type.upper()

        if isinstance(latent_model_type, str):
            latent_model_type = latent_model_type.strip()
            if latent_model_type.lower() in {"", "none"}:
                latent_model_type = None

        if isinstance(feature_model_type, str):
            feature_model_type = feature_model_type.strip()
            if feature_model_type.lower() in {"", "none"}:
                feature_model_type = None
            else:
                feature_model_type = feature_model_type.upper()
        if isinstance(synthetic_geography, str):
            synthetic_geography = synthetic_geography.strip().lower()
        if isinstance(ppa_contract, str):
            ppa_contract = ppa_contract.strip().lower()
        if isinstance(synthetic_ppa_cluster_mode, str):
            synthetic_ppa_cluster_mode = synthetic_ppa_cluster_mode.strip().lower()
        if isinstance(synthetic_congestion_weight_shape, str):
            synthetic_congestion_weight_shape = (
                synthetic_congestion_weight_shape.strip().lower()
            )
        if isinstance(synthetic_model_b_features, str):
            synthetic_model_b_features = synthetic_model_b_features.strip().lower()
        if isinstance(synthetic_wind_dynamics, str):
            synthetic_wind_dynamics = synthetic_wind_dynamics.strip().lower()
        if isinstance(synthetic_local_curtailment_mode, str):
            synthetic_local_curtailment_mode = (
                synthetic_local_curtailment_mode.strip().lower()
            )

        if latent_model_type is None:
            if legacy_model_type in [None, "A", "B", "C"]:
                latent_model_type = "era5field"
            elif legacy_model_type in ["Replication", "replication"]:
                latent_model_type = "replication"
            elif legacy_model_type in [
                "synthetic_field",
                "synthethic_field",
                "synthetic",
            ]:
                latent_model_type = "synthetic"
            else:
                raise ValueError(
                    f"Unknown legacy model_type '{legacy_model_type}'. "
                    "Use latent_model_type in {'replication', 'synthetic', 'era5field'}."
                )
        else:
            latent_model_type = latent_model_type.lower()

        use_exact_sigmoid_expectation = config(
            "use_exact_sigmoid_expectation",
            latent_model_type == "replication",
            bool,
            help=(
                "Use Gauss-Hermite integration for E[sigmoid(X_T)] instead of "
                "the legacy call-spread approximation. Enabled by default for "
                "the Biegler-Koenig replication setup to avoid artificial drift."
            ),
        )

        if feature_model_type is None:
            if legacy_model_type in ["A", "B", "C"]:
                feature_model_type = legacy_model_type
            elif legacy_model_type in ["synthetic_field", "synthethic_field"]:
                feature_model_type = "C"
            else:
                feature_model_type = "A"

        valid_latent_model_types = {"replication", "synthetic", "era5field"}
        valid_feature_model_types = {"A", "B", "C"}
        valid_synthetic_geographies = {"single_country", "cross_border"}
        valid_ppa_contracts = {
            "portfolio",
            "local_cluster",
            "local_cluster_congestion",
            "nl_single",
            "nl_single_capture",
            "local_portfolio_capture",
            "nl_spatial_simple",
        }
        if latent_model_type not in valid_latent_model_types:
            raise ValueError(
                f"Invalid latent_model_type '{latent_model_type}'. Expected one of "
                f"{sorted(valid_latent_model_types)}."
            )
        if feature_model_type not in valid_feature_model_types:
            raise ValueError(
                f"Invalid feature_model_type '{feature_model_type}'. Expected one of "
                f"{sorted(valid_feature_model_types)}."
            )
        if synthetic_geography not in valid_synthetic_geographies:
            raise ValueError(
                f"Invalid synthetic_geography '{synthetic_geography}'. Expected one of "
                f"{sorted(valid_synthetic_geographies)}."
            )
        if ppa_contract not in valid_ppa_contracts:
            raise ValueError(
                f"Invalid ppa_contract '{ppa_contract}'. Expected one of "
                f"{sorted(valid_ppa_contracts)}."
            )
        valid_synthetic_ppa_cluster_modes = {
            "explicit",
            "legacy",
            "mixed_low_loading",
            "mixed_high",
            "mixed_extreme",
            "congestion_high",
            "single_high",
        }
        if synthetic_ppa_cluster_mode not in valid_synthetic_ppa_cluster_modes:
            raise ValueError(
                f"Invalid synthetic_ppa_cluster_mode '{synthetic_ppa_cluster_mode}'. "
                f"Expected one of {sorted(valid_synthetic_ppa_cluster_modes)}."
            )
        valid_congestion_weight_shapes = {"linear", "exponential"}
        if synthetic_congestion_weight_shape not in valid_congestion_weight_shapes:
            raise ValueError(
                f"Invalid synthetic_congestion_weight_shape "
                f"'{synthetic_congestion_weight_shape}'. Expected one of "
                f"{sorted(valid_congestion_weight_shapes)}."
            )
        valid_synthetic_model_b_features = {"regional", "aggregate_dispersion"}
        if synthetic_model_b_features not in valid_synthetic_model_b_features:
            raise ValueError(
                f"Invalid synthetic_model_b_features '{synthetic_model_b_features}'. "
                f"Expected one of {sorted(valid_synthetic_model_b_features)}."
            )
        valid_synthetic_wind_dynamics = {
            "ou_spatial",
            "minimal_volume_risk",
            "two_region_block_ou",
            "two_region_factor_sites",
        }
        if synthetic_wind_dynamics not in valid_synthetic_wind_dynamics:
            raise ValueError(
                f"Invalid synthetic_wind_dynamics '{synthetic_wind_dynamics}'. "
                f"Expected one of {sorted(valid_synthetic_wind_dynamics)}."
            )
        if synthetic_wind_dynamics in {"two_region_block_ou", "two_region_factor_sites"}:
            _log.verify(
                synthetic_num_sites % 2 == 0,
                "synthetic_wind_dynamics='%s' requires an even number of sites; "
                "found %d.",
                synthetic_wind_dynamics,
                synthetic_num_sites,
            )
            _log.verify(
                -0.99 < synthetic_two_region_cross_corr < 1.0,
                "'synthetic_two_region_cross_corr' must be in (-0.99, 1); found %g.",
                synthetic_two_region_cross_corr,
            )
            _log.verify(
                -0.99 < synthetic_two_region_within_corr < 1.0,
                "'synthetic_two_region_within_corr' must be in (-0.99, 1); found %g.",
                synthetic_two_region_within_corr,
            )
            _log.verify(
                synthetic_site_noise_vol >= 0.0,
                "'synthetic_site_noise_vol' must be non-negative; found %g.",
                synthetic_site_noise_vol,
            )
        valid_synthetic_local_curtailment_modes = {
            "regional_pro_rata",
            "contract_cap",
        }
        if synthetic_local_curtailment_mode not in valid_synthetic_local_curtailment_modes:
            raise ValueError(
                "Invalid synthetic_local_curtailment_mode "
                f"'{synthetic_local_curtailment_mode}'. Expected one of "
                f"{sorted(valid_synthetic_local_curtailment_modes)}."
            )
        _log.verify(
            ppa_capacity_scale > 0.0,
            "'ppa_capacity_scale' must be positive; found %g",
            ppa_capacity_scale,
        )

        is_cross_border_synthetic = (
            latent_model_type == "synthetic" and synthetic_geography == "cross_border"
        )
        is_single_country_local_cluster = (
            latent_model_type == "synthetic"
            and synthetic_geography == "single_country"
            and ppa_contract == "local_cluster"
        )
        is_single_country_local_cluster_congestion = (
            latent_model_type == "synthetic"
            and synthetic_geography == "single_country"
            and ppa_contract == "local_cluster_congestion"
        )
        is_single_country_local_cluster_family = (
            is_single_country_local_cluster or is_single_country_local_cluster_congestion
        )
        is_minimal_volume_risk_synthetic = (
            latent_model_type == "synthetic"
            and synthetic_geography == "single_country"
            and synthetic_wind_dynamics == "minimal_volume_risk"
        )
        if is_minimal_volume_risk_synthetic:
            _log.verify(
                is_single_country_local_cluster_congestion,
                "synthetic_wind_dynamics='minimal_volume_risk' is only supported "
                "with ppa_contract='local_cluster_congestion'.",
            )
            _log.verify(
                0 < synthetic_minimal_congested_sites < synthetic_num_sites,
                "'synthetic_minimal_congested_sites' must be between 1 and "
                "synthetic_num_sites - 1. Found %d for %d sites.",
                synthetic_minimal_congested_sites,
                synthetic_num_sites,
            )
        is_nl_spatial_simple = is_cross_border_synthetic and (
            ppa_contract == "nl_spatial_simple"
        )
        site_country_codes = None
        site_type_codes = None

        if latent_model_type == "replication":
            num_dim = 2
            k_vals = np.array([kappa_q1, kappa_q2])
            s_vals = np.array([sigma_q1, sigma_q2])
            targets = np.array([q1_target, q2_target])
            model_weights = np.array([w1, w2])

            persistence = np.diag(np.exp(-k_vals * dt))
            cov_matrix = np.array(
                [
                    [sigma_q1**2, rho * sigma_q1 * sigma_q2],
                    [rho * sigma_q1 * sigma_q2, sigma_q2**2],
                ]
            )
            discrete_cov = cov_matrix * (
                (1 - np.exp(-2 * kappa_q1 * dt)) / (2 * kappa_q1)
            )
            vol_step = np.linalg.cholesky(discrete_cov)

        elif latent_model_type == "synthetic":
            if is_cross_border_synthetic:
                num_dim = 12
                site_country_codes = np.array([0] * 6 + [1] * 6, dtype=int)
                site_type_codes = np.array([0, 0, 0, 1, 1, 1] * 2, dtype=int)

                coords = []
                for country_idx in range(2):
                    x_base = country_idx * synthetic_country_spacing
                    for site_idx in range(3):
                        coords.append([x_base + site_idx * synthetic_spacing, 0.0])
                    for site_idx in range(3):
                        coords.append(
                            [
                                x_base + site_idx * synthetic_spacing,
                                synthetic_offshore_offset,
                            ]
                        )
                coords = np.array(coords, dtype=self.np_dtype)
                dist_matrix = cdist(coords, coords, metric="euclidean")
                corr_matrix = np.exp(-dist_matrix / synthetic_length_scale)

                k_vals = np.where(
                    site_type_codes == 0,
                    synthetic_mean_reversion_onshore,
                    synthetic_mean_reversion_offshore,
                ).astype(self.np_dtype)
                s_vals = np.where(
                    site_type_codes == 0, synthetic_vol_onshore, synthetic_vol_offshore
                ).astype(self.np_dtype)
                targets = np.where(
                    site_type_codes == 0,
                    synthetic_target_onshore,
                    synthetic_target_offshore,
                ).astype(self.np_dtype)
                model_weights = np.full(num_dim, 1.0 / num_dim, dtype=self.np_dtype)

                if is_nl_spatial_simple:
                    # Keep the latent-field mathematics intact, but break the NL offshore
                    # symmetry so Model C can learn from composition rather than totals only.
                    nl_onshore_idx = np.where(
                        (site_country_codes == 0) & (site_type_codes == 0)
                    )[0]
                    nl_offshore_idx = np.where(
                        (site_country_codes == 0) & (site_type_codes == 1)
                    )[0]
                    targets[nl_onshore_idx] += np.array(
                        [-0.03, 0.00, 0.03], dtype=self.np_dtype
                    )
                    targets[nl_offshore_idx] += np.array(
                        [-0.06, 0.00, 0.06], dtype=self.np_dtype
                    )
                    k_vals[nl_offshore_idx] *= np.array(
                        [0.90, 1.00, 1.10], dtype=self.np_dtype
                    )
                    s_vals[nl_offshore_idx] *= np.array(
                        [0.90, 1.00, 1.10], dtype=self.np_dtype
                    )
                    targets = np.clip(targets, 0.05, 0.95).astype(self.np_dtype)

                persistence = np.diag(np.exp(-k_vals * dt))

                step_scales = s_vals * np.sqrt(
                    (1 - np.exp(-2 * k_vals * dt)) / (2 * k_vals)
                )
                vol_step = np.diag(step_scales) @ np.linalg.cholesky(corr_matrix)
            else:
                num_dim = synthetic_num_sites
                if is_single_country_local_cluster_family:
                    if is_minimal_volume_risk_synthetic:
                        # Minimal volume-risk world:
                        # first block is the congestion-prone PPA region, second
                        # block is unconstrained. We keep the existing local-cluster
                        # machinery by coding congested sites as "coastal" and
                        # unconstrained sites as "inland".
                        n_coastal = synthetic_minimal_congested_sites
                        n_inland = num_dim - n_coastal
                        n_offshore = 0
                        site_type_codes = np.array(
                            ([1] * n_coastal) + ([0] * n_inland),
                            dtype=int,
                        )
                        coords = np.array(
                            [[i * synthetic_spacing, 0.0] for i in range(num_dim)],
                            dtype=self.np_dtype,
                        )
                        k_vals = np.full(
                            num_dim, synthetic_mean_reversion, dtype=self.np_dtype
                        )
                        s_vals = np.full(num_dim, synthetic_vol, dtype=self.np_dtype)
                        targets = np.full(
                            num_dim, synthetic_target, dtype=self.np_dtype
                        )
                        model_weights = np.ones(num_dim, dtype=self.np_dtype) / num_dim
                        corr_matrix = np.eye(num_dim, dtype=self.np_dtype)
                        persistence = np.diag(np.exp(-k_vals * dt))
                        step_scales = s_vals * np.sqrt(
                            (1 - np.exp(-2 * k_vals * dt)) / (2 * k_vals)
                        )
                        vol_step = np.diag(step_scales)
                    else:
                        # Three German wind groups: inland onshore, coastal onshore, offshore.
                        # Unlike the pure spatial-kernel setup, this branch uses a low-rank
                        # latent factor model with site loadings so the contracted cluster is
                        # not too well summarized by the country aggregate.
                        n_offshore = max(2, num_dim // 3)
                        n_coastal = max(2, (num_dim - n_offshore) // 2)
                        n_inland = num_dim - n_coastal - n_offshore
                        site_type_codes = np.array(
                            ([0] * n_inland) + ([1] * n_coastal) + ([2] * n_offshore),
                            dtype=int,
                        )
                        coords = []
                        for i in range(n_inland):
                            coords.append([i * synthetic_spacing, 0.0])
                        for i in range(n_coastal):
                            coords.append(
                                [i * synthetic_spacing, synthetic_offshore_offset * 0.6]
                            )
                        for i in range(n_offshore):
                            coords.append(
                                [i * synthetic_spacing, synthetic_offshore_offset * 1.4]
                            )
                        coords = np.array(coords, dtype=self.np_dtype)

                        k_vals = np.full(
                            num_dim, synthetic_mean_reversion, dtype=self.np_dtype
                        )
                        s_vals = np.full(num_dim, synthetic_vol, dtype=self.np_dtype)
                        targets = np.full(num_dim, synthetic_target, dtype=self.np_dtype)
                        model_weights = np.ones(num_dim, dtype=self.np_dtype) / num_dim

                        inland_idx = np.where(site_type_codes == 0)[0]
                        coastal_idx = np.where(site_type_codes == 1)[0]
                        offshore_idx = np.where(site_type_codes == 2)[0]
                        targets[inland_idx] += np.linspace(-0.05, 0.01, len(inland_idx), dtype=self.np_dtype)
                        targets[coastal_idx] += np.linspace(0.02, 0.06, len(coastal_idx), dtype=self.np_dtype)
                        targets[offshore_idx] += np.linspace(0.05, 0.09, len(offshore_idx), dtype=self.np_dtype)
                        s_vals[coastal_idx] *= 1.05
                        s_vals[offshore_idx] *= 1.15
                        k_vals[coastal_idx] *= 0.95
                        k_vals[offshore_idx] *= 0.9
                        targets = np.clip(targets, 0.05, 0.95).astype(self.np_dtype)

                        # Latent factors: national, coastal, offshore.
                        factor_loadings = np.zeros((num_dim, 3), dtype=self.np_dtype)
                        if len(inland_idx) > 0:
                            factor_loadings[inland_idx, 0] = 1.00
                            factor_loadings[inland_idx, 1] = np.linspace(
                                0.05, 0.15, len(inland_idx), dtype=self.np_dtype
                            )
                            factor_loadings[inland_idx, 2] = np.linspace(
                                0.00, 0.05, len(inland_idx), dtype=self.np_dtype
                            )
                        if len(coastal_idx) > 0:
                            factor_loadings[coastal_idx, 0] = 0.90
                            factor_loadings[coastal_idx, 1] = np.linspace(
                                0.60, 0.85, len(coastal_idx), dtype=self.np_dtype
                            )
                            factor_loadings[coastal_idx, 2] = np.linspace(
                                0.10, 0.20, len(coastal_idx), dtype=self.np_dtype
                            )
                        if len(offshore_idx) > 0:
                            factor_loadings[offshore_idx, 0] = 0.75
                            factor_loadings[offshore_idx, 1] = np.linspace(
                                0.20, 0.35, len(offshore_idx), dtype=self.np_dtype
                            )
                            factor_loadings[offshore_idx, 2] = np.linspace(
                                0.75, 1.00, len(offshore_idx), dtype=self.np_dtype
                            )

                        factor_cov = factor_loadings @ factor_loadings.T
                        idio_var = synthetic_local_cluster_idio**2
                        cov_matrix = factor_cov + idio_var * np.eye(
                            num_dim, dtype=self.np_dtype
                        )
                        std_vec = np.sqrt(np.diag(cov_matrix))
                        corr_matrix = cov_matrix / np.outer(std_vec, std_vec)
                        # Keep a mild national floor so the whole country still co-moves.
                        corr_matrix = synthetic_national_corr + (
                            1.0 - synthetic_national_corr
                        ) * corr_matrix
                        np.fill_diagonal(corr_matrix, 1.0)

                        persistence = np.diag(np.exp(-k_vals * dt))
                        step_scales = s_vals * np.sqrt(
                            (1 - np.exp(-2 * k_vals * dt)) / (2 * k_vals)
                        )
                        vol_step = np.diag(step_scales) @ np.linalg.cholesky(corr_matrix)
                else:
                    coords_cong = [[i * 10.0, 0.0] for i in range(num_dim // 2)]
                    coords_unc = [
                        [500 + i * 10.0, 0.0] for i in range(num_dim - (num_dim // 2))
                    ]
                    coords = np.array(coords_cong + coords_unc)
                    k_vals = np.full(num_dim, synthetic_mean_reversion)
                    s_vals = np.full(num_dim, synthetic_vol)
                    targets = np.full(num_dim, synthetic_target)
                    model_weights = np.ones(num_dim) / num_dim
                    if synthetic_wind_dynamics == "two_region_block_ou":
                        split = num_dim // 2
                        corr_matrix = np.full(
                            (num_dim, num_dim),
                            synthetic_two_region_cross_corr,
                            dtype=self.np_dtype,
                        )
                        corr_matrix[:split, :split] = synthetic_two_region_within_corr
                        corr_matrix[split:, split:] = synthetic_two_region_within_corr
                        np.fill_diagonal(corr_matrix, 1.0)
                        min_eig = float(np.min(np.linalg.eigvalsh(corr_matrix)))
                        if min_eig <= 1e-8:
                            corr_matrix += np.eye(num_dim, dtype=self.np_dtype) * (
                                1e-8 - min_eig
                            )
                    elif synthetic_wind_dynamics == "two_region_factor_sites":
                        split = num_dim // 2
                        within_corr = float(synthetic_two_region_within_corr)
                        cross_corr = float(synthetic_two_region_cross_corr)
                        _log.verify(
                            0.0 < within_corr < 1.0,
                            "'two_region_factor_sites' requires "
                            "synthetic_two_region_within_corr in (0, 1); found %g.",
                            within_corr,
                        )
                        _log.verify(
                            abs(cross_corr) < within_corr,
                            "'two_region_factor_sites' requires |cross_corr| < "
                            "within_corr; found cross=%g, within=%g.",
                            cross_corr,
                            within_corr,
                        )
                        # Two latent regional OU factors drive most variation.
                        # Independent site noise makes the ten site forecasts
                        # distinct while preserving the requested within/cross
                        # site correlations.
                        idio_var = max(float(synthetic_site_noise_vol) ** 2, 1e-12)
                        factor_var = idio_var * within_corr / (1.0 - within_corr)
                        factor_corr = cross_corr / within_corr
                        factor_cov = factor_var * np.array(
                            [[1.0, factor_corr], [factor_corr, 1.0]],
                            dtype=self.np_dtype,
                        )
                        factor_loadings = np.zeros((num_dim, 2), dtype=self.np_dtype)
                        factor_loadings[:split, 0] = 1.0
                        factor_loadings[split:, 1] = 1.0
                        cov_matrix = (
                            factor_loadings @ factor_cov @ factor_loadings.T
                            + idio_var * np.eye(num_dim, dtype=self.np_dtype)
                        )
                        std_vec = np.sqrt(np.diag(cov_matrix))
                        corr_matrix = cov_matrix / np.outer(std_vec, std_vec)
                        np.fill_diagonal(corr_matrix, 1.0)
                    else:
                        dist_matrix = cdist(coords, coords, metric="euclidean")
                        corr_matrix = np.exp(-dist_matrix / synthetic_length_scale)

                    congestion_flow_weights = np.ones(num_dim, dtype=self.np_dtype)
                    # Site-specific loading weights make congestion depend on composition,
                    # not just on the congested-region aggregate.
                    congestion_flow_weights[: num_dim // 2] = np.linspace(
                        1.0 - synthetic_congestion_weight_spread,
                        1.0 + synthetic_congestion_weight_spread,
                        num_dim // 2,
                        dtype=self.np_dtype,
                    )
                    if synthetic_wind_dynamics not in {"two_region_block_ou", "two_region_factor_sites"}:
                        targets[: num_dim // 2] += np.linspace(
                            -0.04,
                            0.04,
                            num_dim // 2,
                            dtype=self.np_dtype,
                        )
                        s_vals[: num_dim // 2] *= np.linspace(
                            0.9,
                            1.1,
                            num_dim // 2,
                            dtype=self.np_dtype,
                        )
                    persistence = np.diag(np.exp(-k_vals * dt))
                    step_scales = s_vals * np.sqrt(
                        (1 - np.exp(-2 * k_vals * dt)) / (2 * k_vals)
                    )
                    vol_step = np.diag(step_scales) @ np.linalg.cholesky(corr_matrix)

            if save_synthetic_diagnostics:
                plt.figure(figsize=(10, 8))
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    cmap="coolwarm",
                    fmt=".2f",
                    vmin=0,
                    vmax=1,
                )
                plt.title(
                    f"Synthethic spatial correlation (length scale: {synthetic_length_scale})"
                )
                plt.xlabel("West -> East")
                plt.ylabel("West -> East")
                plt.savefig("corr_matrix_synthetic.jpg", dpi=300)
                plt.close()

                plt.figure(figsize=(10, 8))
                sns.heatmap(vol_step, annot=True, cmap="viridis", fmt=".3f")
                plt.title("Cholesky decomposition (Source of noise)")
                plt.xlabel("Indepdent Noise (z)")
                plt.ylabel("Location Index (Q)")
                plt.savefig("vol_step_synthetic.jpg", dpi=300)
                plt.close()

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

        if is_nl_spatial_simple or is_single_country_local_cluster_family:
            # Major weather-forecast arrivals for the paper-like spatial test worlds.
            update_times = [0, 10, 14, 18, 34, 38, 42]
        else:
            update_times = list(range(nSteps + 1))
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

        if is_cross_border_synthetic:
            nl_mask = site_country_codes == 0
            de_mask = site_country_codes == 1
            nl_onshore_mask = nl_mask & (site_type_codes == 0)
            nl_offshore_mask = nl_mask & (site_type_codes == 1)
            if is_nl_spatial_simple:
                q_onshore_total_0 = np.mean(E_q_0[nl_onshore_mask])
                q_offshore_total_0 = np.mean(E_q_0[nl_offshore_mask])
                g0_nl = 1.0 - (w1 * q_onshore_total_0 + w2 * q_offshore_total_0)
                g0_de = None
            else:
                g0_nl = 1.0 - synthetic_price_beta_nl * np.sum(
                    model_weights[nl_mask] * E_q_0[nl_mask]
                )
                g0_de = 1.0 - synthetic_price_beta_de * np.sum(
                    model_weights[de_mask] * E_q_0[de_mask]
                )
        else:
            if is_single_country_local_cluster_family:
                inland_mask = site_type_codes == 0
                coastal_mask = site_type_codes == 1
                offshore_mask = site_type_codes == 2
                g0 = 1.0 - np.sum(model_weights * E_q_0)
                inland_indices = np.where(inland_mask)[0]
                coastal_indices = np.where(coastal_mask)[0]
                offshore_indices = np.where(offshore_mask)[0]
                if len(ppa_cluster_indices_override) > 0:
                    ppa_cluster_indices = np.asarray(
                        ppa_cluster_indices_override, dtype=int
                    )
                elif synthetic_ppa_cluster_mode == "explicit":
                    raise ValueError(
                        "synthetic_ppa_cluster_mode='explicit' requires "
                        "non-empty ppa_cluster_indices."
                    )
                elif synthetic_ppa_cluster_mode == "legacy":
                    ppa_cluster_indices = np.concatenate(
                        [coastal_indices[:2], offshore_indices[:1]]
                    )
                elif synthetic_ppa_cluster_mode == "mixed_high":
                    ppa_cluster_indices = np.array(
                        [
                            inland_indices[len(inland_indices) // 2],
                            coastal_indices[-1],
                            offshore_indices[-1],
                        ],
                        dtype=int,
                    )
                elif synthetic_ppa_cluster_mode == "mixed_low_loading":
                    ppa_cluster_indices = np.array(
                        [
                            inland_indices[len(inland_indices) // 2],
                            coastal_indices[0],
                            offshore_indices[0],
                        ],
                        dtype=int,
                    )
                elif synthetic_ppa_cluster_mode == "mixed_extreme":
                    ppa_cluster_indices = np.array(
                        [inland_indices[0], coastal_indices[-1], offshore_indices[-1]],
                        dtype=int,
                    )
                elif synthetic_ppa_cluster_mode == "congestion_high":
                    ppa_cluster_indices = np.concatenate(
                        [coastal_indices[-1:], offshore_indices[-2:]]
                    )
                else:
                    ppa_cluster_indices = np.array([offshore_indices[-1]], dtype=int)
                ppa_cluster_indices = np.unique(ppa_cluster_indices).astype(int)
                _log.verify(
                    len(ppa_cluster_indices) > 0,
                    "PPA cluster must contain at least one site.",
                )
                _log.verify(
                    np.all((ppa_cluster_indices >= 0) & (ppa_cluster_indices < num_dim)),
                    "PPA cluster indices %s are outside the valid site range [0, %d).",
                    ppa_cluster_indices,
                    num_dim,
                )
                if len(ppa_contract_weights_override) > 0:
                    ppa_contract_weights = np.asarray(
                        ppa_contract_weights_override, dtype=self.np_dtype
                    )
                    _log.verify(
                        len(ppa_contract_weights) == len(ppa_cluster_indices),
                        "ppa_contract_weights must have the same length as "
                        "ppa_cluster_indices. Found %d weights for %d sites.",
                        len(ppa_contract_weights),
                        len(ppa_cluster_indices),
                    )
                    _log.verify(
                        np.all(ppa_contract_weights >= 0.0),
                        "ppa_contract_weights must be non-negative.",
                    )
                    _log.verify(
                        np.sum(ppa_contract_weights) > 0.0,
                        "ppa_contract_weights must have positive total weight.",
                    )
                else:
                    ppa_contract_weights = model_weights[ppa_cluster_indices].astype(
                        self.np_dtype, copy=False
                    )
                ppa_cluster_weights = ppa_capacity_scale * ppa_contract_weights
                ppa_site_weights = np.zeros(num_dim, dtype=self.np_dtype)
                ppa_site_weights[ppa_cluster_indices] = ppa_cluster_weights
                congestion_region_mask = coastal_mask | offshore_mask
                congestion_region_indices = np.where(congestion_region_mask)[0]
                ppa_cluster_congestion_mask = np.isin(
                    ppa_cluster_indices, congestion_region_indices
                )
                if synthetic_congestion_weight_shape == "exponential":
                    congestion_flow_weights_local = np.exp(
                        np.linspace(
                            -synthetic_congestion_weight_spread,
                            synthetic_congestion_weight_spread,
                            len(congestion_region_indices),
                            dtype=self.np_dtype,
                        )
                    )
                    congestion_flow_weights_local = (
                        congestion_flow_weights_local
                        / np.mean(congestion_flow_weights_local)
                    ).astype(self.np_dtype)
                else:
                    congestion_flow_weights_local = np.linspace(
                        1.0 - synthetic_congestion_weight_spread,
                        1.0 + synthetic_congestion_weight_spread,
                        len(congestion_region_indices),
                        dtype=self.np_dtype,
                    )
            else:
                g0 = 1.0 - np.sum(model_weights * E_q_0)

        num_dim = len(targets)
        x = np.zeros((nSamples, nSteps + 1, num_dim))
        q_forecasts = np.zeros((nSamples, nSteps + 1, num_dim))

        if is_cross_border_synthetic:
            xp_shared = np.zeros((nSamples, nSteps + 1))
            xp_nl = np.zeros((nSamples, nSteps + 1))
            xp_de = np.zeros((nSamples, nSteps + 1))
            f_t_T_nl = np.zeros((nSamples, nSteps + 1))
            f_t_T_de = np.zeros((nSamples, nSteps + 1))
        else:
            xp = np.zeros((nSamples, nSteps + 1))
            f_t_T = np.zeros((nSamples, nSteps + 1))

        if is_minimal_volume_risk_synthetic:
            n_cong_minimal = synthetic_minimal_congested_sites
            national_factor = np.random.normal(size=(nSamples, 1))
            composition_factor = np.random.normal(size=(nSamples, 1))
            idio_factor = np.random.normal(size=(nSamples, num_dim))
            terminal_latent = np.empty((nSamples, num_dim), dtype=self.np_dtype)
            terminal_latent[:, :n_cong_minimal] = (
                synthetic_minimal_national_loading * national_factor
                + synthetic_minimal_composition_loading * composition_factor
                + synthetic_minimal_idio_loading * idio_factor[:, :n_cong_minimal]
            )
            terminal_latent[:, n_cong_minimal:] = (
                synthetic_minimal_national_loading * national_factor
                - synthetic_minimal_composition_loading * composition_factor
                + synthetic_minimal_idio_loading * idio_factor[:, n_cong_minimal:]
            )
            q_realized_sites = 1.0 / (1.0 + np.exp(-terminal_latent))
            q_total_realized_minimal = np.sum(
                model_weights * q_realized_sites, axis=1
            )
            terminal_price_shock = (
                synthetic_minimal_price_shock_vol * np.random.normal(size=nSamples)
            )

            for t in range(nSteps + 1):
                if t == 0:
                    q_forecasts[:, t, :] = synthetic_target
                elif t == nSteps:
                    q_forecasts[:, t, :] = q_realized_sites
                else:
                    reveal = (t / nSteps) ** 0.75
                    noise_scale = np.sqrt(max(1.0 - reveal**2, 0.0))
                    signal = (
                        reveal * terminal_latent
                        + noise_scale * np.random.normal(size=(nSamples, num_dim))
                    )
                    q_forecasts[:, t, :] = 1.0 / (1.0 + np.exp(-signal))

                q_total_forecast_t = np.sum(model_weights * q_forecasts[:, t, :], axis=1)
                f_t_T[:, t] = (
                    f_0_T
                    - synthetic_minimal_price_wind_beta
                    * (q_total_forecast_t - synthetic_target)
                )

            # The terminal price uses realized full potential national wind plus
            # the unobserved demand/residual-load shock. Congestion never enters
            # the price equation.
            f_t_T[:, -1] = (
                f_0_T
                + terminal_price_shock
                - synthetic_minimal_price_wind_beta
                * (q_total_realized_minimal - synthetic_target)
            )
        else:
            for t in range(nSteps + 1):
                t_curr = t * dt

                if t_curr in update_times:
                    t_minus_idx = t

                T_rem = T_max - (t_minus_idx * dt)

                if t > 0:
                    z = np.random.normal(size=(nSamples, num_dim))
                    x[:, t, :] = (x[:, t - 1, :] @ persistence) + (z @ vol_step.T)

                    if is_cross_border_synthetic:
                        z_shared = np.random.normal(size=nSamples)
                        z_country = np.random.normal(size=(nSamples, 2))
                        z_country[:, 1] = (
                            synthetic_country_price_corr * z_country[:, 0]
                            + np.sqrt(1.0 - synthetic_country_price_corr**2)
                            * z_country[:, 1]
                        )
                        xp_shared[:, t] = (
                            xp_shared[:, t - 1] * np.exp(-kappa_p_shared * dt)
                            + sigma_p_shared
                            * np.sqrt(
                                (1 - np.exp(-2 * kappa_p_shared * dt))
                                / (2 * kappa_p_shared)
                            )
                            * z_shared
                        )
                        xp_nl[:, t] = (
                            xp_nl[:, t - 1] * np.exp(-kappa_p_country * dt)
                            + sigma_p_country
                            * np.sqrt(
                                (1 - np.exp(-2 * kappa_p_country * dt))
                                / (2 * kappa_p_country)
                            )
                            * z_country[:, 0]
                        )
                        xp_de[:, t] = (
                            xp_de[:, t - 1] * np.exp(-kappa_p_country * dt)
                            + sigma_p_country
                            * np.sqrt(
                                (1 - np.exp(-2 * kappa_p_country * dt))
                                / (2 * kappa_p_country)
                            )
                            * z_country[:, 1]
                        )
                    else:
                        zp = np.random.normal(size=nSamples)
                        xp[:, t] = (
                            xp[:, t - 1] * np.exp(-kappa_p * dt)
                            + sigma_p
                            * np.sqrt((1 - np.exp(-2 * kappa_p * dt)) / (2 * kappa_p))
                            * zp
                        )

                eq_m = np.zeros((nSamples, num_dim))
                for i in range(num_dim):
                    eq_m[:, i] = get_expected_q(
                        x[:, t_minus_idx, i], T_rem, k_vals[i], s_vals[i], phis[i]
                    )
                q_forecasts[:, t, :] = eq_m

                if is_cross_border_synthetic:
                    if is_nl_spatial_simple:
                        q_onshore_total_t = np.mean(eq_m[:, nl_onshore_mask], axis=1)
                        q_offshore_total_t = np.mean(eq_m[:, nl_offshore_mask], axis=1)
                        gt_nl = 1.0 - (w1 * q_onshore_total_t + w2 * q_offshore_total_t)
                    else:
                        q_nl_forecast_t = np.sum(
                            model_weights[nl_mask] * eq_m[:, nl_mask], axis=1
                        )
                        q_de_forecast_t = np.sum(
                            model_weights[de_mask] * eq_m[:, de_mask], axis=1
                        )
                        gt_nl = 1.0 - synthetic_price_beta_nl * q_nl_forecast_t
                        gt_de = 1.0 - synthetic_price_beta_de * q_de_forecast_t

                    mt_shared, _ = ou_stats(
                        xp_shared[:, t], kappa_p_shared, sigma_p_shared, t_curr, T_max
                    )
                    mt_nl, _ = ou_stats(
                        xp_nl[:, t], kappa_p_country, sigma_p_country, t_curr, T_max
                    )
                    mt_de, _ = ou_stats(
                        xp_de[:, t], kappa_p_country, sigma_p_country, t_curr, T_max
                    )
                    f_t_T_nl[:, t] = f_0_T_nl * (1.0 + mt_shared + mt_nl) * (gt_nl / g0_nl)
                    if not is_nl_spatial_simple:
                        f_t_T_de[:, t] = (
                            f_0_T_de
                            * (1.0 + mt_shared + mt_de)
                            * (gt_de / g0_de)
                        )
                else:
                    gt = 1.0 - np.sum(model_weights * eq_m, axis=1)
                    mt_mean, _ = ou_stats(xp[:, t], kappa_p, sigma_p, t_curr, T_max)
                    f_t_T[:, t] = f_0_T * (1.0 + mt_mean) * (gt / g0)

            q_realized_sites = 1.0 / (1.0 + np.exp(-(x[:, -1, :] + phis)))
        mwh_realized_sites = q_realized_sites * model_weights

        if is_cross_border_synthetic:
            q_nl_realized = np.sum(mwh_realized_sites[:, nl_mask], axis=1)
            q_de_realized = np.sum(mwh_realized_sites[:, de_mask], axis=1)
            q_nl_delivered = np.minimum(q_nl_realized, l_max_nl)
            q_de_delivered = np.minimum(q_de_realized, l_max_de)
            q_nl_onshore_realized = np.sum(
                mwh_realized_sites[:, nl_onshore_mask], axis=1
            )
            q_nl_offshore_realized = np.sum(
                mwh_realized_sites[:, nl_offshore_mask], axis=1
            )
            q_de_onshore_realized = np.sum(
                mwh_realized_sites[:, de_mask & (site_type_codes == 0)], axis=1
            )
            q_de_offshore_realized = np.sum(
                mwh_realized_sites[:, de_mask & (site_type_codes == 1)], axis=1
            )
            q_nl_delivery_scale = np.divide(
                q_nl_delivered,
                q_nl_realized,
                out=np.zeros_like(q_nl_delivered),
                where=q_nl_realized > 1e-12,
            )
            q_nl_onshore_delivered = q_nl_onshore_realized * q_nl_delivery_scale
            q_nl_offshore_delivered = q_nl_offshore_realized * q_nl_delivery_scale
            q_nl_offshore_share_realized = np.divide(
                q_nl_offshore_realized,
                q_nl_realized,
                out=np.zeros_like(q_nl_offshore_realized),
                where=q_nl_realized > 1e-12,
            )
            q_total_realized = q_nl_realized + q_de_realized
            q_tilde = q_nl_delivered + q_de_delivered

            q_nl_forecast = np.sum(
                model_weights[nl_mask][np.newaxis, np.newaxis, :]
                * q_forecasts[:, :, nl_mask],
                axis=2,
            )
            q_de_forecast = np.sum(
                model_weights[de_mask][np.newaxis, np.newaxis, :]
                * q_forecasts[:, :, de_mask],
                axis=2,
            )
            # Local cross-border PPA cluster: one onshore and one offshore site per country.
            ppa_nl_onshore_idx = np.where(nl_onshore_mask)[0][:1]
            ppa_nl_offshore_idx = np.where(nl_offshore_mask)[0][:1]
            ppa_de_onshore_idx = np.where(de_mask & (site_type_codes == 0))[0][:1]
            ppa_de_offshore_idx = np.where(de_mask & (site_type_codes == 1))[0][:1]
            ppa_local_nl_idx = np.concatenate([ppa_nl_onshore_idx, ppa_nl_offshore_idx])
            ppa_local_de_idx = np.concatenate([ppa_de_onshore_idx, ppa_de_offshore_idx])
            q_ppa_nl_forecast = np.sum(
                model_weights[ppa_local_nl_idx][np.newaxis, np.newaxis, :]
                * q_forecasts[:, :, ppa_local_nl_idx],
                axis=2,
            )
            q_ppa_de_forecast = np.sum(
                model_weights[ppa_local_de_idx][np.newaxis, np.newaxis, :]
                * q_forecasts[:, :, ppa_local_de_idx],
                axis=2,
            )
            q_ppa_nl_onshore_realized = np.sum(
                mwh_realized_sites[:, ppa_nl_onshore_idx], axis=1
            )
            q_ppa_nl_offshore_realized = np.sum(
                mwh_realized_sites[:, ppa_nl_offshore_idx], axis=1
            )
            q_ppa_de_onshore_realized = np.sum(
                mwh_realized_sites[:, ppa_de_onshore_idx], axis=1
            )
            q_ppa_de_offshore_realized = np.sum(
                mwh_realized_sites[:, ppa_de_offshore_idx], axis=1
            )
            q_ppa_nl_realized = q_ppa_nl_onshore_realized + q_ppa_nl_offshore_realized
            q_ppa_de_realized = q_ppa_de_onshore_realized + q_ppa_de_offshore_realized
            if is_nl_spatial_simple:
                spatial_site_mask = nl_mask
                ppa_cluster_indices = np.where(nl_onshore_mask)[0][:2]
                # The PPA cluster is a small local onshore exposure; offshore only moves price.
                q_ppa_forecast = np.mean(q_forecasts[:, :, ppa_cluster_indices], axis=2)
                q_ppa_realized = np.mean(q_realized_sites[:, ppa_cluster_indices], axis=1)
                q_onshore_total_forecast = np.mean(
                    q_forecasts[:, :, nl_onshore_mask], axis=2
                )
                q_offshore_total_forecast = np.mean(
                    q_forecasts[:, :, nl_offshore_mask], axis=2
                )
                q_onshore_total_realized = np.mean(
                    q_realized_sites[:, nl_onshore_mask], axis=1
                )
                q_offshore_total_realized = np.mean(
                    q_realized_sites[:, nl_offshore_mask], axis=1
                )

                path_history = np.zeros((nSamples, nSteps + 1, 4), dtype=self.np_dtype)
                path_history[:, :, 0] = f_t_T_nl
                path_history[:, :-1, 1] = q_ppa_forecast[:, :-1]
                path_history[:, :-1, 2] = q_onshore_total_forecast[:, :-1]
                path_history[:, :-1, 3] = q_offshore_total_forecast[:, :-1]
                path_history[:, -1, 1] = q_ppa_realized
                path_history[:, -1, 2] = q_onshore_total_realized
                path_history[:, -1, 3] = q_offshore_total_realized

                payoff = q_ppa_realized * (f_t_T_nl[:, -1] - ppa_strike_nl)
                p_capture = (
                    np.mean(
                        f_t_T_nl[:, -1][:, np.newaxis]
                        * q_realized_sites[:, spatial_site_mask],
                        axis=0,
                    )
                    / np.mean(q_realized_sites[:, spatial_site_mask], axis=0)
                )
                cannibal_rat = p_capture / f_0_T_nl

                dInsts = np.zeros((nSamples, nSteps, 1), dtype=self.np_dtype)
                dInsts[:, :, 0] = (
                    f_t_T_nl[:, -1][:, np.newaxis] - f_t_T_nl[:, :-1]
                )
            else:
                path_history = np.zeros((nSamples, nSteps + 1, 6), dtype=self.np_dtype)
                path_history[:, :, 0] = f_t_T_nl
                path_history[:, :, 1] = f_t_T_de
                path_history[:, :-1, 2] = q_nl_forecast[:, :-1]
                path_history[:, :-1, 3] = q_de_forecast[:, :-1]
                path_history[:, -1, 2] = q_nl_realized
                path_history[:, -1, 3] = q_de_realized
                path_history[:, :, 4] = q_nl_delivered[:, np.newaxis]
                path_history[:, :, 5] = q_de_delivered[:, np.newaxis]

                payoff_nl = q_nl_delivered * (f_t_T_nl[:, -1] - ppa_strike_nl)
                payoff_de = q_de_delivered * (f_t_T_de[:, -1] - ppa_strike_de)
                nl_onshore_capture_price = (
                    ppa_capture_beta_onshore_nl_forward * f_t_T_nl[:, -1]
                    + ppa_capture_beta_onshore_de_forward * f_t_T_de[:, -1]
                    - ppa_capture_discount_nl_onshore
                )
                nl_offshore_capture_price = (
                    ppa_capture_beta_offshore_nl_forward * f_t_T_nl[:, -1]
                    + ppa_capture_beta_offshore_de_forward * f_t_T_de[:, -1]
                    - ppa_capture_discount_nl_offshore
                    - ppa_capture_cannibalization_nl_offshore
                    * q_nl_offshore_realized
                )
                de_onshore_capture_price = (
                    ppa_capture_beta_onshore_de_forward * f_t_T_nl[:, -1]
                    + ppa_capture_beta_onshore_nl_forward * f_t_T_de[:, -1]
                    - ppa_capture_discount_nl_onshore
                )
                de_offshore_capture_price = (
                    ppa_capture_beta_offshore_de_forward * f_t_T_nl[:, -1]
                    + ppa_capture_beta_offshore_nl_forward * f_t_T_de[:, -1]
                    - ppa_capture_discount_nl_offshore
                    - ppa_capture_cannibalization_nl_offshore
                    * q_de_offshore_realized
                )
                payoff_nl_capture = (
                    q_nl_onshore_delivered
                    * (nl_onshore_capture_price - ppa_strike_nl)
                    + q_nl_offshore_delivered
                    * (nl_offshore_capture_price - ppa_strike_nl)
                )
                payoff_local_portfolio_capture = (
                    q_ppa_nl_onshore_realized
                    * (nl_onshore_capture_price - ppa_strike_nl)
                    + q_ppa_nl_offshore_realized
                    * (nl_offshore_capture_price - ppa_strike_nl)
                    + q_ppa_de_onshore_realized
                    * (de_onshore_capture_price - ppa_strike_de)
                    + q_ppa_de_offshore_realized
                    * (de_offshore_capture_price - ppa_strike_de)
                )
                if ppa_contract == "nl_single":
                    payoff = payoff_nl
                elif ppa_contract == "nl_single_capture":
                    payoff = payoff_nl_capture
                elif ppa_contract == "local_portfolio_capture":
                    # Local cross-border payoff keeps the hedge tradeable with NL/DE forwards
                    # while making site composition matter through onshore/offshore betas.
                    payoff = payoff_local_portfolio_capture
                else:
                    payoff = payoff_nl + payoff_de

                terminal_price_by_site = np.where(
                    nl_mask[np.newaxis, :],
                    f_t_T_nl[:, -1][:, np.newaxis],
                    f_t_T_de[:, -1][:, np.newaxis],
                )
                p_capture = np.mean(
                    terminal_price_by_site * q_realized_sites, axis=0
                ) / np.mean(q_realized_sites, axis=0)
                cannibal_rat = p_capture / np.where(
                    nl_mask, f_0_T_nl, f_0_T_de
                )

                dInsts = np.zeros((nSamples, nSteps, 2), dtype=self.np_dtype)
                dInsts[:, :, 0] = f_t_T_nl[:, -1][:, np.newaxis] - f_t_T_nl[:, :-1]
                dInsts[:, :, 1] = f_t_T_de[:, -1][:, np.newaxis] - f_t_T_de[:, :-1]
        else:
            q_total_realized = np.sum(model_weights * q_realized_sites, axis=1)
            path_history = np.zeros((nSamples, nSteps + 1, 3), dtype=self.np_dtype)
            path_history[:, :, 0] = f_t_T
            q_agg_forecast = np.sum(model_weights * q_forecasts, axis=2)
            q_cross_sectional_dispersion_forecast = np.sqrt(
                np.sum(
                    model_weights[np.newaxis, np.newaxis, :]
                    * (q_forecasts - q_agg_forecast[:, :, np.newaxis]) ** 2,
                    axis=2,
                )
            )

            if latent_model_type == "replication":
                # Biegler-Koenig replication mode: the national forward price
                # is driven by weighted onshore/offshore infeed, while the PPA
                # itself is written on the onshore wind asset only.
                q_tilde = q_realized_sites[:, 0]
                path_history[:, :-1, 1] = q_forecasts[:, :-1, 0]
                path_history[:, -1, 1] = q_tilde
                payoff = ppa_payoff(path_history, q_tilde)

                q_cong = mwh_realized_sites[:, 0]
                q_unc = (
                    mwh_realized_sites[:, 1]
                    if num_dim > 1
                    else np.zeros_like(q_total_realized)
                )
                q_cong_flow = q_cong
                curtailment_ratio = np.ones_like(q_total_realized)
                congestion_flow_weights = np.ones(num_dim, dtype=self.np_dtype)
                q_cong_forecast = model_weights[0] * q_forecasts[:, :, 0]
                q_unc_forecast = (
                    model_weights[1] * q_forecasts[:, :, 1]
                    if num_dim > 1
                    else np.zeros_like(q_agg_forecast)
                )
            elif is_single_country_local_cluster_family:
                q_inland_forecast = np.sum(
                    model_weights[inland_mask] * q_forecasts[:, :, inland_mask], axis=2
                )
                q_coastal_forecast = np.sum(
                    model_weights[coastal_mask] * q_forecasts[:, :, coastal_mask], axis=2
                )
                q_offshore_forecast = np.sum(
                    model_weights[offshore_mask] * q_forecasts[:, :, offshore_mask], axis=2
                )
                ppa_cluster_cong_indices = ppa_cluster_indices[
                    ppa_cluster_congestion_mask
                ]
                ppa_cluster_unc_indices = ppa_cluster_indices[
                    ~ppa_cluster_congestion_mask
                ]
                q_cluster_forecast = np.sum(
                    ppa_cluster_weights * q_forecasts[:, :, ppa_cluster_indices],
                    axis=2,
                )
                if len(ppa_cluster_cong_indices) > 0:
                    q_cluster_cong_forecast = np.sum(
                        ppa_site_weights[ppa_cluster_cong_indices]
                        * q_forecasts[:, :, ppa_cluster_cong_indices],
                        axis=2,
                    )
                    q_cluster_cong_realized = np.sum(
                        ppa_site_weights[ppa_cluster_cong_indices]
                        * q_realized_sites[:, ppa_cluster_cong_indices],
                        axis=1,
                    )
                else:
                    q_cluster_cong_forecast = np.zeros(
                        (nSamples, nSteps + 1), dtype=self.np_dtype
                    )
                    q_cluster_cong_realized = np.zeros(nSamples, dtype=self.np_dtype)
                if len(ppa_cluster_unc_indices) > 0:
                    q_cluster_unc_forecast = np.sum(
                        ppa_site_weights[ppa_cluster_unc_indices]
                        * q_forecasts[:, :, ppa_cluster_unc_indices],
                        axis=2,
                    )
                    q_cluster_unc_realized = np.sum(
                        ppa_site_weights[ppa_cluster_unc_indices]
                        * q_realized_sites[:, ppa_cluster_unc_indices],
                        axis=1,
                    )
                else:
                    q_cluster_unc_forecast = np.zeros(
                        (nSamples, nSteps + 1), dtype=self.np_dtype
                    )
                    q_cluster_unc_realized = np.zeros(nSamples, dtype=self.np_dtype)
                q_inland_realized = np.sum(mwh_realized_sites[:, inland_mask], axis=1)
                q_coastal_realized = np.sum(mwh_realized_sites[:, coastal_mask], axis=1)
                q_offshore_realized = np.sum(mwh_realized_sites[:, offshore_mask], axis=1)
                q_cluster_realized = np.sum(
                    ppa_cluster_weights * q_realized_sites[:, ppa_cluster_indices],
                    axis=1,
                )
                q_congestion_region_realized = np.sum(
                    mwh_realized_sites[:, congestion_region_mask], axis=1
                )
                q_congestion_region_forecast = np.sum(
                    model_weights[congestion_region_mask]
                    * q_forecasts[:, :, congestion_region_mask],
                    axis=2,
                )
                q_cross_sectional_dispersion_forecast = np.sqrt(
                    np.sum(
                        model_weights[np.newaxis, np.newaxis, :]
                        * (
                            q_forecasts
                            - q_agg_forecast[:, :, np.newaxis]
                        )
                        ** 2,
                        axis=2,
                    )
                )

                path_history = np.zeros((nSamples, nSteps + 1, 5), dtype=self.np_dtype)
                path_history[:, :, 0] = f_t_T
                path_history[:, :-1, 1] = q_agg_forecast[:, :-1]
                path_history[:, :-1, 2] = q_inland_forecast[:, :-1]
                path_history[:, :-1, 3] = q_coastal_forecast[:, :-1]
                path_history[:, :-1, 4] = q_offshore_forecast[:, :-1]
                path_history[:, -1, 1] = q_total_realized
                path_history[:, -1, 2] = q_inland_realized
                path_history[:, -1, 3] = q_coastal_realized
                path_history[:, -1, 4] = q_offshore_realized

                if is_single_country_local_cluster_congestion:
                    # Country price still sees potential generation, while cluster payout
                    # depends on delivered volume after regional congestion.
                    q_regional_congestion_flow = np.sum(
                        congestion_flow_weights_local[np.newaxis, :]
                        * mwh_realized_sites[:, congestion_region_indices],
                        axis=1,
                    )
                    if synthetic_local_curtailment_mode == "contract_cap":
                        # Literal volume-risk setup:
                        # Q_tilde = min(Q_ppa_cong, L_max) + Q_ppa_unc.
                        # Here L_max is calibrated on the contracted congested
                        # PPA volume, not on the full regional grid flow.
                        q_congestion_flow = q_cluster_cong_realized
                        q_cluster_cong_delivered = np.minimum(
                            q_cluster_cong_realized, l_max
                        )
                        congestion_curtailment_ratio = np.divide(
                            q_cluster_cong_delivered,
                            q_cluster_cong_realized,
                            out=np.ones_like(q_cluster_cong_realized),
                            where=q_cluster_cong_realized > 1e-12,
                        )
                        q_cluster_delivered = (
                            q_cluster_cong_delivered + q_cluster_unc_realized
                        )
                    else:
                        # Previous interpretation: the whole congested region
                        # is curtailed pro-rata, and contracted sites inherit
                        # the same regional curtailment ratio.
                        q_congestion_flow = q_regional_congestion_flow
                        congestion_curtailment_ratio = np.minimum(
                            1.0,
                            np.divide(
                                l_max,
                                q_congestion_flow,
                                out=np.ones_like(q_congestion_flow),
                                where=q_congestion_flow > 1e-12,
                            ),
                        )
                        q_cluster_delivered = (
                            congestion_curtailment_ratio * q_cluster_cong_realized
                            + q_cluster_unc_realized
                        )
                    payoff = q_cluster_delivered * (f_t_T[:, -1] - f_0_T)
                    q_tilde = q_cluster_delivered
                else:
                    q_cluster_delivered = q_cluster_realized
                    q_congestion_flow = np.zeros_like(q_cluster_realized)
                    q_regional_congestion_flow = np.zeros_like(q_cluster_realized)
                    congestion_curtailment_ratio = np.ones_like(q_cluster_realized)
                    payoff = q_cluster_realized * (f_t_T[:, -1] - f_0_T)
                    q_tilde = q_cluster_realized
            else:
                # First half of the 1D grid is the congestion-prone export region.
                q_cong = mwh_realized_sites[:, : num_dim // 2].sum(axis=1)
                q_unc = mwh_realized_sites[:, num_dim // 2 : num_dim].sum(axis=1)
                q_cong_flow = np.sum(
                    congestion_flow_weights[: num_dim // 2]
                    * mwh_realized_sites[:, : num_dim // 2],
                    axis=1,
                )
                curtailment_ratio = np.minimum(
                    1.0,
                    np.divide(
                        l_max,
                        q_cong_flow,
                        out=np.ones_like(q_cong_flow),
                        where=q_cong_flow > 1e-12,
                    ),
                )
                q_tilde = curtailment_ratio * q_cong + q_unc
                q_cong_forecast = np.sum(
                    model_weights[: num_dim // 2] * q_forecasts[:, :, : num_dim // 2],
                    axis=2,
                )
                q_unc_forecast = np.sum(
                    model_weights[num_dim // 2 :] * q_forecasts[:, :, num_dim // 2 :],
                    axis=2,
                )
                path_history[:, :-1, 1] = q_agg_forecast[:, :-1]
                path_history[:, -1, 1] = q_total_realized
                payoff = ppa_payoff(path_history, q_tilde)

            f_T = f_t_T[:, -1]
            p_capture = (
                np.mean(f_T[:, np.newaxis] * q_realized_sites, axis=0)
                / np.mean(q_realized_sites, axis=0)
            )
            cannibal_rat = p_capture / f_0_T

            dInsts = np.zeros((nSamples, nSteps, 1), dtype=self.np_dtype)
            dInsts[:, :, 0] = f_t_T[:, nSteps][:, np.newaxis] - f_t_T[:, :nSteps]

        spread_base = 0.01
        spread_max = 0.03
        tau = time_left
        lamb = 5.0
        bid_ask_spread_free = spread_base + spread_max * np.exp(-lamb * tau)
        cost_mwh = 0.15

        if is_cross_border_synthetic:
            if is_nl_spatial_simple:
                cost_matrix = (
                    f_t_T_nl[:, :nSteps] * (bid_ask_spread_free / 2.0)
                ) + cost_mwh
                if use_transaction_cost:
                    cost = cost_matrix[:, :, np.newaxis].astype(
                        self.np_dtype, copy=False
                    )
                else:
                    cost = np.zeros((nSamples, nSteps, 1), dtype=self.np_dtype)
                price = f_t_T_nl[:, :nSteps]
                ubnd_a = np.full((nSamples, nSteps, 1), ubnd_a, dtype=self.np_dtype)
                lbnd_a = np.full((nSamples, nSteps, 1), lbnd_a, dtype=self.np_dtype)
                ubnd_trade = np.full((nSamples, nSteps, 1), ubnd_trade, dtype=self.np_dtype)
                lbnd_trade = np.full((nSamples, nSteps, 1), lbnd_trade, dtype=self.np_dtype)
            else:
                cost_matrix = np.zeros((nSamples, nSteps, 2), dtype=self.np_dtype)
                cost_matrix[:, :, 0] = (
                    f_t_T_nl[:, :nSteps] * (bid_ask_spread_free / 2.0)
                ) + cost_mwh
                cost_matrix[:, :, 1] = (
                    f_t_T_de[:, :nSteps] * (bid_ask_spread_free / 2.0)
                ) + cost_mwh
                if use_transaction_cost:
                    cost = cost_matrix.astype(self.np_dtype, copy=False)
                else:
                    cost = np.zeros((nSamples, nSteps, 2), dtype=self.np_dtype)
                price = np.stack([f_t_T_nl[:, :nSteps], f_t_T_de[:, :nSteps]], axis=-1)
                ubnd_a = np.full((nSamples, nSteps, 2), ubnd_a, dtype=self.np_dtype)
                lbnd_a = np.full((nSamples, nSteps, 2), lbnd_a, dtype=self.np_dtype)
                ubnd_trade = np.full((nSamples, nSteps, 2), ubnd_trade, dtype=self.np_dtype)
                lbnd_trade = np.full((nSamples, nSteps, 2), lbnd_trade, dtype=self.np_dtype)
        else:
            cost_matrix = (f_t_T[:, :nSteps] * (bid_ask_spread_free / 2.0)) + cost_mwh
            if use_transaction_cost:
                cost = cost_matrix[:, :, np.newaxis].astype(self.np_dtype, copy=False)
            else:
                cost = np.zeros((nSamples, nSteps, 1), dtype=self.np_dtype)
            price = f_t_T[:, :nSteps]
            ubnd_a = np.full((nSamples, nSteps, 1), ubnd_a, dtype=self.np_dtype)
            lbnd_a = np.full((nSamples, nSteps, 1), lbnd_a, dtype=self.np_dtype)
            ubnd_trade = np.full((nSamples, nSteps, 1), ubnd_trade, dtype=self.np_dtype)
            lbnd_trade = np.full((nSamples, nSteps, 1), lbnd_trade, dtype=self.np_dtype)

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
            hedges=dInsts,
            cost=cost,
            ubnd_a=ubnd_a,
            lbnd_a=lbnd_a,
            ubnd_trade=ubnd_trade,
            lbnd_trade=lbnd_trade,
            payoff=payoff,
        )

        # features
        # observable variables for the agent
        per_step_features = pdct(
            time_left=np.full(
                (nSamples, nSteps), time_left[np.newaxis, :], dtype=self.np_dtype
            ),
            forward_price=price,
            cost=cost,
        )
        if is_cross_border_synthetic:
            if is_nl_spatial_simple:
                if feature_model_type == "A":
                    # Model A gets grouped aggregate wind information only.
                    per_step_features.wind_info = np.stack(
                        [
                            q_ppa_forecast[:, :-1],
                            q_onshore_total_forecast[:, :-1],
                            q_offshore_total_forecast[:, :-1],
                        ],
                        axis=-1,
                    )
                elif feature_model_type == "B":
                    offshore_dispersion = np.var(
                        q_forecasts[:, :-1, nl_offshore_mask], axis=2
                    )
                    per_step_features.wind_info = np.stack(
                        [
                            q_ppa_forecast[:, :-1],
                            q_onshore_total_forecast[:, :-1],
                            q_offshore_total_forecast[:, :-1],
                            offshore_dispersion,
                        ],
                        axis=-1,
                    )
                elif feature_model_type == "C":
                    # Model C sees the full disaggregated NL spatial field.
                    per_step_features.wind_info = q_forecasts[:, :-1, nl_mask]
            else:
                if feature_model_type == "A":
                    if ppa_contract in {"nl_single", "nl_single_capture"}:
                        per_step_features.wind_info = q_nl_forecast[:, :-1, np.newaxis]
                    elif ppa_contract == "local_portfolio_capture":
                        # Model A only sees country-level aggregates, not the local asset mix.
                        per_step_features.wind_info = np.stack(
                            [q_nl_forecast[:, :-1], q_de_forecast[:, :-1]], axis=-1
                        )
                    else:
                        per_step_features.wind_info = np.stack(
                            [q_nl_forecast[:, :-1], q_de_forecast[:, :-1]], axis=-1
                        )
                elif feature_model_type == "B":
                    dispersion_nl = np.var(q_forecasts[:, :-1, nl_mask], axis=2)
                    dispersion_de = np.var(q_forecasts[:, :-1, de_mask], axis=2)
                    if ppa_contract in {"nl_single", "nl_single_capture"}:
                        per_step_features.wind_info = np.stack(
                            [q_nl_forecast[:, :-1], dispersion_nl],
                            axis=-1,
                        )
                    elif ppa_contract == "local_portfolio_capture":
                        # Model B sees local country-level cluster forecasts, but not the
                        # onshore/offshore composition inside the cluster.
                        per_step_features.wind_info = np.stack(
                            [
                                q_nl_forecast[:, :-1],
                                q_de_forecast[:, :-1],
                                q_ppa_nl_forecast[:, :-1],
                                q_ppa_de_forecast[:, :-1],
                            ],
                            axis=-1,
                        )
                    else:
                        per_step_features.wind_info = np.stack(
                            [
                                q_nl_forecast[:, :-1],
                                q_de_forecast[:, :-1],
                                dispersion_nl,
                                dispersion_de,
                            ],
                            axis=-1,
                        )
                elif feature_model_type == "C":
                    per_step_features.wind_info = q_forecasts[:, :-1, :]
        else:
            if is_single_country_local_cluster_family:
                if feature_model_type == "A":
                    per_step_features.wind_info = q_agg_forecast[:, :-1, np.newaxis]
                elif feature_model_type == "B":
                    if synthetic_model_b_features == "aggregate_dispersion":
                        # Aggregate-plus-dispersion benchmark: B knows whether the
                        # field is concentrated, but not where it is concentrated.
                        per_step_features.wind_info = np.stack(
                            [
                                q_agg_forecast[:, :-1],
                                q_cross_sectional_dispersion_forecast[:, :-1],
                            ],
                            axis=-1,
                        )
                    else:
                        # Model B gets grouped regional aggregates but not the full site panel.
                        per_step_features.wind_info = np.stack(
                            [
                                q_inland_forecast[:, :-1],
                                q_coastal_forecast[:, :-1],
                                q_offshore_forecast[:, :-1],
                            ],
                            axis=-1,
                        )
                elif feature_model_type == "C":
                    per_step_features.wind_info = q_forecasts[:, :-1, :]
            else:
                if feature_model_type == "A":
                    per_step_features.wind_info = q_agg_forecast[:, :-1, np.newaxis]
                elif feature_model_type == "B":
                    if synthetic_model_b_features == "aggregate_dispersion":
                        # Fair benchmark for the spatial volume-risk mechanism: B knows
                        # concentration but not which region is producing it.
                        per_step_features.wind_info = np.stack(
                            [
                                q_agg_forecast[:, :-1],
                                q_cross_sectional_dispersion_forecast[:, :-1],
                            ],
                            axis=-1,
                        )
                    else:
                        # Stronger legacy benchmark: B sees block-level congestion
                        # information, which partly reveals the hidden split.
                        per_step_features.wind_info = np.stack(
                            [q_cong_forecast[:, :-1], q_unc_forecast[:, :-1]], axis=-1
                        )
                elif feature_model_type == "C":
                    per_step_features.wind_info = q_forecasts[:, :-1, :]
        
        self.data.features = pdct(per_step=per_step_features, per_path=pdct())

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
        if is_cross_border_synthetic:
            if is_nl_spatial_simple:
                self.details = pdct(
                    forward_price=f_t_T_nl,
                    q_ppa_forecast=q_ppa_forecast,
                    q_onshore_total_forecast=q_onshore_total_forecast,
                    q_offshore_total_forecast=q_offshore_total_forecast,
                    q_ppa_realized=q_ppa_realized,
                    q_onshore_total_realized=q_onshore_total_realized,
                    q_offshore_total_realized=q_offshore_total_realized,
                    q_total_realized=q_onshore_total_realized + q_offshore_total_realized,
                    payoff=payoff,
                    path_history=path_history,
                    cannibal_rat=cannibal_rat,
                    q_tilde=q_ppa_realized,
                )
            else:
                self.details = pdct(
                    forward_price=np.stack([f_t_T_nl, f_t_T_de], axis=-1),
                    forward_price_nl=f_t_T_nl,
                    forward_price_de=f_t_T_de,
                    q_nl_forecast=q_nl_forecast,
                    q_de_forecast=q_de_forecast,
                    q_nl_realized=q_nl_realized,
                    q_de_realized=q_de_realized,
                    q_nl_delivered=q_nl_delivered,
                    q_de_delivered=q_de_delivered,
                    q_nl_onshore_realized=q_nl_onshore_realized,
                    q_nl_offshore_realized=q_nl_offshore_realized,
                    q_de_onshore_realized=q_de_onshore_realized,
                    q_de_offshore_realized=q_de_offshore_realized,
                    q_nl_onshore_delivered=q_nl_onshore_delivered,
                    q_nl_offshore_delivered=q_nl_offshore_delivered,
                    q_nl_offshore_share_realized=q_nl_offshore_share_realized,
                    q_total_realized=q_total_realized,
                    payoff=payoff,
                    payoff_nl=payoff_nl,
                    payoff_de=payoff_de,
                    payoff_nl_capture=payoff_nl_capture,
                    payoff_local_portfolio_capture=payoff_local_portfolio_capture,
                    nl_onshore_capture_price=nl_onshore_capture_price,
                    nl_offshore_capture_price=nl_offshore_capture_price,
                    de_onshore_capture_price=de_onshore_capture_price,
                    de_offshore_capture_price=de_offshore_capture_price,
                    q_ppa_nl_forecast=q_ppa_nl_forecast,
                    q_ppa_de_forecast=q_ppa_de_forecast,
                    q_ppa_nl_realized=q_ppa_nl_realized,
                    q_ppa_de_realized=q_ppa_de_realized,
                    q_ppa_nl_onshore_realized=q_ppa_nl_onshore_realized,
                    q_ppa_nl_offshore_realized=q_ppa_nl_offshore_realized,
                    q_ppa_de_onshore_realized=q_ppa_de_onshore_realized,
                    q_ppa_de_offshore_realized=q_ppa_de_offshore_realized,
                    ppa_local_nl_idx=ppa_local_nl_idx,
                    ppa_local_de_idx=ppa_local_de_idx,
                    nl_offshore_cannibalization_discount=(
                        ppa_capture_cannibalization_nl_offshore
                        * q_nl_offshore_realized
                    ),
                    path_history=path_history,
                    cannibal_rat=cannibal_rat,
                    q_tilde=q_tilde,
                )
        else:
            if is_single_country_local_cluster_family:
                self.details = pdct(
                    forward_price=f_t_T,
                    q_agg_forecast=q_agg_forecast,
                    q_inland_forecast=q_inland_forecast,
                    q_coastal_forecast=q_coastal_forecast,
                    q_offshore_forecast=q_offshore_forecast,
                    q_cluster_forecast=q_cluster_forecast,
                    q_cluster_cong_forecast=q_cluster_cong_forecast,
                    q_cluster_unc_forecast=q_cluster_unc_forecast,
                    q_congestion_region_forecast=q_congestion_region_forecast,
                    q_cross_sectional_dispersion_forecast=q_cross_sectional_dispersion_forecast,
                    q_total_realized=q_total_realized,
                    q_inland_realized=q_inland_realized,
                    q_coastal_realized=q_coastal_realized,
                    q_offshore_realized=q_offshore_realized,
                    q_cluster_realized=q_cluster_realized,
                    q_cluster_cong_realized=q_cluster_cong_realized,
                    q_cluster_unc_realized=q_cluster_unc_realized,
                    q_cluster_delivered=q_cluster_delivered,
                    q_congestion_region_realized=q_congestion_region_realized,
                    q_congestion_flow=q_congestion_flow,
                    q_regional_congestion_flow=q_regional_congestion_flow,
                    q_congestion_curtailment_ratio=1.0 - congestion_curtailment_ratio,
                    payoff=payoff,
                    path_history=path_history,
                    cannibal_rat=cannibal_rat,
                    q_tilde=q_tilde,
                    ppa_cluster_indices=ppa_cluster_indices,
                    ppa_cluster_cong_indices=ppa_cluster_cong_indices,
                    ppa_cluster_unc_indices=ppa_cluster_unc_indices,
                    ppa_cluster_congestion_mask=ppa_cluster_congestion_mask.astype(int),
                    ppa_contract_weights=ppa_contract_weights,
                    ppa_cluster_weights=ppa_cluster_weights,
                    ppa_site_weights=ppa_site_weights,
                    ppa_capacity_scale=np.array(ppa_capacity_scale),
                    congestion_region_indices=congestion_region_indices,
                    congestion_flow_weights=congestion_flow_weights_local,
                )
            else:
                self.details = pdct(
                    forward_price=f_t_T,
                    q_agg_forecast=q_agg_forecast,
                    q_cong_forecast=q_cong_forecast,
                    q_unc_forecast=q_unc_forecast,
                    q_cross_sectional_dispersion_forecast=q_cross_sectional_dispersion_forecast,
                    q_total_realized=q_total_realized,
                    payoff=payoff,
                    path_history=path_history,
                    cannibal_rat=cannibal_rat,
                    q_tilde=q_tilde,
                    q_cong=q_cong,
                    q_cong_flow=q_cong_flow,
                    q_unc=q_unc,
                    q_curtailment=(
                        np.zeros_like(q_tilde)
                        if latent_model_type == "replication"
                        else q_total_realized - q_tilde
                    ),
                    q_curtailment_ratio=(
                        np.zeros_like(q_tilde)
                        if latent_model_type == "replication"
                        else 1.0 - curtailment_ratio
                    ),
                    q_cluster_realized=(
                        q_tilde if latent_model_type == "replication" else q_total_realized
                    ),
                    q_cluster_delivered=q_tilde,
                    q_congestion_region_realized=(
                        np.zeros_like(q_tilde)
                        if latent_model_type == "replication"
                        else q_cong
                    ),
                    q_congestion_flow=(
                        np.zeros_like(q_tilde)
                        if latent_model_type == "replication"
                        else q_cong_flow
                    ),
                    q_congestion_curtailment_ratio=(
                        np.zeros_like(q_tilde)
                        if latent_model_type == "replication"
                        else 1.0 - curtailment_ratio
                    ),
                    congestion_flow_weights=(
                        np.zeros(0, dtype=self.np_dtype)
                        if latent_model_type == "replication"
                        else congestion_flow_weights[: num_dim // 2]
                    ),
                    ppa_cluster_indices=(
                        np.array([0], dtype=int)
                        if latent_model_type == "replication"
                        else np.arange(num_dim, dtype=int)
                    ),
                    ppa_contract_weights=(
                        np.array([1.0], dtype=self.np_dtype)
                        if latent_model_type == "replication"
                        else model_weights.astype(self.np_dtype, copy=False)
                    ),
                    ppa_capacity_scale=np.array(1.0),
                    congestion_region_indices=(
                        np.zeros(0, dtype=int)
                        if latent_model_type == "replication"
                        else np.arange(num_dim // 2, dtype=int)
                    ),
                    q_cong_ratio_realized=np.divide(
                        0.0 if latent_model_type == "replication" else q_cong,
                        q_total_realized,
                        out=np.zeros_like(q_cong),
                        where=q_total_realized > 1e-12,
                    ),
                )
        # Keep details strictly numeric so assert_iter_not_is_nan can validate it.
        if num_dim > 1:
            if is_nl_spatial_simple:
                self.details.site_forecasts = q_forecasts[:, :, nl_mask]
                self.details.site_realized = q_realized_sites[:, nl_mask]
            else:
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

        if is_cross_border_synthetic:
            if is_nl_spatial_simple:
                self.inst_names = ["NL Forward"]
                self.site_country_codes = np.zeros(np.sum(nl_mask), dtype=int)
                self.site_type_codes = site_type_codes[nl_mask]
                self.site_country_names = ["NL"]
                self.site_type_names = ["onshore", "offshore"]
                self.ppa_cluster_site_indices = np.array([0, 1], dtype=int)
            else:
                self.inst_names = ["NL Forward", "DE Forward"]
                self.site_country_codes = site_country_codes
                self.site_type_codes = site_type_codes
                self.site_country_names = ["NL", "DE"]
                self.site_type_names = ["onshore", "offshore"]
        else:
            self.inst_names = ["DE Forward"]
            if is_single_country_local_cluster_family:
                self.site_type_codes = site_type_codes
                self.site_type_names = ["inland_onshore", "coastal_onshore", "offshore"]
        # Static and dynamic volume hedging
        
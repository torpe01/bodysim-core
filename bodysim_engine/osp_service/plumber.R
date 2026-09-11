# plumber.R — BodySim OSP Bridge REST Service
#
# Wraps ospsuite-R to expose a PK-Sim PBPK simulation as a single REST
# endpoint.  The Aciclovir.pkml bundled with ospsuite is used as a
# structural template; all compound parameters are overwritten at
# runtime from the JSON request body produced by admet.py.
#
# Start the service:
#   Rscript -e "pr <- plumber::plumb('osp_service/plumber.R'); pr$run(port=8000, host='0.0.0.0')"
#
# POST /simulate   — run simulation, return C(t) + NCA metrics
# GET  /health     — confirm service is alive

library(plumber)
library(ospsuite)
library(jsonlite)

# ── Constants ────────────────────────────────────────────────────────────────
PKML_TEMPLATE <- system.file("extdata/Aciclovir.pkml", package = "ospsuite")
PLASMA_PATH   <- paste0(
  "Organism|PeripheralVenousBlood|Aciclovir|",
  "Plasma (Peripheral Venous Blood)"
)
DRUG_PREFIX   <- "Aciclovir|"

# ── Null-coalescing operator ─────────────────────────────────────────────────
`%||%` <- function(a, b) if (!is.null(a) && length(a) > 0) a else b

# ── Unit conversion helpers ──────────────────────────────────────────────────

# OSP time axis is in minutes; PK convention uses hours
min_to_h <- function(x) x / 60.0

# OSP concentration is in µmol/L; clinical PK uses mg/L
# C [mg/L] = C [µmol/L] × MW [g/mol] / 1000
umol_to_mgl <- function(x, mw_g_per_mol) x * mw_g_per_mol / 1000.0

# MW: g/mol → kg/µmol (OSP internal unit)
# 1 g/mol = 1e-3 kg / 1e6 µmol = 1e-9 kg/µmol
mw_to_osp <- function(mw_g_per_mol) mw_g_per_mol * 1e-9

# Peff: cm/s → dm/min (OSP internal unit for intestinal permeability)
# 1 cm/s = 0.1 dm/s = 0.1 × 60 dm/min = 6 dm/min
peff_to_osp <- function(peff_cms) peff_cms * 6.0

# ── NCA helpers ─────────────────────────────────────────────────────────────

# Trapezoidal AUC
trap_auc <- function(t, c) {
  n <- length(t)
  if (n < 2) return(0.0)
  sum(diff(t) * (head(c, -1) + tail(c, -1)) / 2.0, na.rm = TRUE)
}

# Log-linear terminal half-life using last 30% of profile
terminal_thalf <- function(t_h, c_mgl) {
  n   <- length(t_h)
  idx <- seq(max(1L, as.integer(0.7 * n)), n)
  t_s <- t_h[idx]
  c_s <- c_mgl[idx]
  c_s[c_s <= 0] <- 1e-12
  fit <- tryCatch(lm(log(c_s) ~ t_s), error = function(e) NULL)
  if (is.null(fit)) return(NA_real_)
  k <- coef(fit)[["t_s"]]
  if (is.na(k) || k >= 0) return(NA_real_)
  -log(2) / k
}

# ── Plumber API ──────────────────────────────────────────────────────────────

#* @apiTitle BodySim OSP Bridge
#* @apiDescription PK-Sim PBPK simulation via ospsuite-R.

#* Health check — confirms service is running and ospsuite is loaded.
#* @get /health
function() {
  list(
    status   = "ok",
    ospsuite = as.character(packageVersion("ospsuite")),
    plumber  = as.character(packageVersion("plumber")),
    template = basename(PKML_TEMPLATE)
  )
}

#* Run a PBPK simulation for a drug described by its physicochemical
#* parameters.  All parameters come from admet.py output.
#*
#* Request body (JSON):
#*   drug_name   : string   — label for logging/response
#*   logp        : float    — lipophilicity (log P)
#*   fup         : float    — fraction unbound in plasma [0-1]
#*   mw          : float    — molecular weight [g/mol]
#*   pka_acid    : float    — acid pKa  (0 if not applicable)
#*   pka_base    : float    — base pKa  (0 if not applicable)
#*   peff_cms    : float    — intestinal permeability [cm/s]
#*   rb          : float    — blood:plasma concentration ratio
#*   dose_mg     : float    — dose [mg]
#*   t_end_h     : float    — simulation duration [hours]  default 24
#*
#* Response body (JSON):
#*   drug_name, status, nca{cmax_mg_l, tmax_h, auc_mg_l_h, t_half_h},
#*   time_h[], conc_mg_l[]
#*
#* @post /simulate
#* @serializer json
function(req, res) {

  # ── Parse body ─────────────────────────────────────────────────────
  body <- tryCatch(
    fromJSON(req$postBody, simplifyVector = TRUE),
    error = function(e) {
      res$status <- 400L
      return(list(error = paste("Invalid JSON:", e$message)))
    }
  )
  if (!is.null(body$error)) return(body)

  drug_name  <- as.character(body$drug_name  %||% "Drug")
  logp       <- as.numeric(body$logp        %||% 0.0)
  fup        <- as.numeric(body$fup         %||% 0.5)
  mw         <- as.numeric(body$mw          %||% 300.0)
  pka_acid   <- as.numeric(body$pka_acid    %||% 0.0)
  pka_base   <- as.numeric(body$pka_base    %||% 0.0)
  peff_cms   <- as.numeric(body$peff_cms    %||% 1e-5)
  rb         <- as.numeric(body$rb          %||% 1.0)
  dose_mg    <- as.numeric(body$dose_mg     %||% 250.0)
  t_end_h    <- as.numeric(body$t_end_h     %||% 24.0)

  # ── Load template (fresh copy each call) ──────────────────────────
  sim <- tryCatch(
    loadSimulation(PKML_TEMPLATE, loadFromCache = FALSE),
    error = function(e) {
      res$status <- 500L
      return(list(error = paste("Failed to load template:", e$message)))
    }
  )
  if (!is.null(sim$error)) return(sim)

  # ── Build parameter vectors ────────────────────────────────────────
  param_paths <- c(
    paste0(DRUG_PREFIX, "Lipophilicity"),
    paste0(DRUG_PREFIX, "Fraction unbound (plasma, reference value)"),
    paste0(DRUG_PREFIX, "Molecular weight"),
    paste0(DRUG_PREFIX, "Specific intestinal permeability (transcellular)"),
    paste0(DRUG_PREFIX, "Blood/Plasma concentration ratio"),
    paste0(DRUG_PREFIX, "pKa value 0"),
    paste0(DRUG_PREFIX, "pKa value 1")
  )

  param_values <- c(
    logp,
    fup,
    mw_to_osp(mw),
    peff_to_osp(peff_cms),
    rb,
    pka_acid,
    pka_base
  )

  # ── Apply parameters ───────────────────────────────────────────────
  tryCatch(
    setParameterValuesByPath(
      values      = param_values,
      parameterPaths      = param_paths,
      simulation      = sim
    ),
    error = function(e) {
      res$status <- 500L
      stop(paste("Parameter error:", e$message))
    }
  )

  # ── Configure output ───────────────────────────────────────────────
  # Keep the existing output intervals from the template but add our path
  tryCatch({
    clearOutputs(sim)
    addOutputs(
      quantitiesOrPaths = PLASMA_PATH,
      simulation        = sim
    )
  }, error = function(e) {
    res$status <- 500L
    stop(paste("Output config error:", e$message))
  })

  # ── Run ────────────────────────────────────────────────────────────
  sim_results <- tryCatch(
    runSimulations(simulations = sim)[[1]],
    error = function(e) {
      res$status <- 500L
      stop(paste("Simulation failed:", e$message))
    }
  )

  # ── Extract C(t) ───────────────────────────────────────────────────
  raw <- tryCatch(
    getOutputValues(
      simulationResults = sim_results,
      quantitiesOrPaths = PLASMA_PATH
    ),
    error = function(e) {
      res$status <- 500L
      stop(paste("Result extraction failed:", e$message))
    }
  )

  t_min  <- raw$data$Time
  c_umol <- raw$data[[PLASMA_PATH]]

  t_h   <- min_to_h(t_min)
  c_mgl <- umol_to_mgl(c_umol, mw)

  # ── NCA ────────────────────────────────────────────────────────────
  cmax   <- max(c_mgl, na.rm = TRUE)
  tmax   <- t_h[which.max(c_mgl)]
  auc    <- trap_auc(t_h, c_mgl)
  t_half <- terminal_thalf(t_h, c_mgl)

  # ── Response — use unbox() so scalars don't become [value] in JSON ──
  list(
    drug_name = jsonlite::unbox(drug_name),
    status    = jsonlite::unbox("success"),
    nca = list(
      cmax_mg_l  = jsonlite::unbox(round(cmax,   6)),
      tmax_h     = jsonlite::unbox(round(tmax,   3)),
      auc_mg_l_h = jsonlite::unbox(round(auc,    4)),
      t_half_h   = if (!is.na(t_half)) jsonlite::unbox(round(t_half, 3)) else NULL
    ),
    time_h    = round(t_h,   3),
    conc_mg_l = round(c_mgl, 8)
  )
}
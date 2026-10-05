#---------------------------------------------------------------------#
#                                                                     #
#   Plasticity rate and capacity, one row per video                   #
#                                                                     #
#   Takes the per-frame L*a*b* values from CIE_conversion.R and, for  #
#   each video, works out how fast and how far the fish changed -     #
#   measured three ways, side by side, so they can be compared:       #
#                                                                     #
#   PRIMARY - straight-line trend in raw L*                           #
#     rate     = slope of L* against time (L* per minute).            #
#                Positive = got lighter, negative = got darker.       #
#     capacity = the fitted change across the trial (slope x length). #
#     Needs no asymptote, is in the same unit for every fish, and     #
#     uses every frame - so one noisy frame can't hijack it.          #
#                                                                     #
#   SECONDARY - the protocol's Dt approach, made robust               #
#     Dt = (zt - z_inf) / (z0 - z_inf), as stated in the protocol,    #
#     but z_inf is the AVERAGE OF THE LAST FEW FRAMES rather than the #
#     single most extreme frame. Linear and exponential shapes are    #
#     both fitted, but the shape is chosen ONCE FOR ALL FISH (the     #
#     protocol fits models "to determine the best fitting shape of    #
#     the curve" - singular), and then applied to every fish, so the  #
#     rate is in the same unit for everyone and can be analysed.      #
#                                                                     #
#   ORIGINAL - the protocol exactly as first implemented              #
#     z_inf = single most extreme frame. Kept for comparison only:    #
#     in the APHP-white pilot it pointed the opposite way from the    #
#     actual trend in 2 of 19 fish, because one noisy frame set it.   #
#                                                                     #
#   Every video also gets a camera column (GoPro / iPhone, from the   #
#   file name) and a flag for recording dates from an unset clock.    #
#                                                                     #
#---------------------------------------------------------------------#

rm(list = ls())
library(dplyr)

#=====================================================================#
#  SETTINGS                                                           #
#=====================================================================#

# Output of CIE_conversion.R (the whole-body average file).
lab_file <- "C:/Users/lamoo/ColorChangeRates2026/files/APHP/white/white_colormaps/LAB_average_white.csv"

# Written by extract_frames.py - carries the time of every frame.
manifest_file <- "C:/Users/lamoo/ColorChangeRates2026/files/APHP/white/white_frames/frames_manifest.csv"

# From find_fish_and_landmarks.py - carries the fish ID you typed.
landmarks_csv <- "C:/Users/lamoo/ColorChangeRates2026/files/APHP/white/white_landmarks/landmarks.csv"

# Created if it doesn't exist.
output_dir <- "C:/Users/lamoo/ColorChangeRates2026/files/APHP/white/results/"

# Which channel to analyse. L* (lightness) for darkening/lightening;
# "a_mean" or "b_mean" run the same analysis on the colour axes.
phenotype_column <- "L_mean"

# A video needs at least this many frames to fit anything.
min_frames <- 4

# SECONDARY analysis: how many frames to average for each end.
#   z_inf is the mean of the LAST z_inf_end_frames frames.
#   z0 is the mean of the FIRST z0_start_frames frames. The protocol
#   defines z0 as the first measurement, so this stays at 1 by default;
#   raise it to 2 to make the baseline as robust as the asymptote.
z_inf_end_frames <- 2
z0_start_frames  <- 1

# The Dt rates divide by (z0 - z_inf). A fish whose change is smaller
# than this many times its measurement noise gets no Dt rate, because Dt
# would be mostly noise divided by noise. 2 is a judgement call - lower it
# to keep more fish, raise it to be stricter. The primary slope is never
# affected by this.
reliability_multiple <- 2

# Curve shape used for EVERY fish in the secondary analysis:
#   "auto"        - chosen once from all fish together (lower total AIC)
#   "linear"      - force a straight line
#   "exponential" - force an exponential approach
curve_shape <- "auto"

# Camera, worked out from the start of each video's file name.
camera_patterns <- c(GoPro = "^(GH|GX|GOPR|GP)", iPhone = "^IMG_")

# Recording dates earlier than this are treated as a camera clock that
# was never set (GoPros default to 2016-01-01).
earliest_plausible_date <- as.Date("2020-01-01")

#=====================================================================#
#  Read and join (unchanged from the version that already ran)        #
#=====================================================================#

dd <- read.csv(lab_file)
cat("Read", nrow(dd), "rows from", basename(lab_file), "\n")

strip_ext <- function(x) sub("\\.[A-Za-z0-9]+$", "", as.character(x))

if (file.exists(manifest_file)) {
  manifest <- read.csv(manifest_file)
  manifest$join_key <- strip_ext(manifest$image)
  keep <- intersect(c("join_key", "video", "video_recorded_at",
                      "video_recorded_date", "video_recorded_time",
                      "video_time_source", "frame_timestamp", "seconds"),
                    names(manifest))
  dd$join_key <- strip_ext(dd[[1]])
  dd <- left_join(dd, manifest[, keep], by = "join_key",
                  suffix = c("", "_manifest"))
  if (!"seconds" %in% names(dd) && "seconds_manifest" %in% names(dd)) {
    dd$seconds <- dd$seconds_manifest
  }
  if (!"video" %in% names(dd) && "video_manifest" %in% names(dd)) {
    dd$video <- dd$video_manifest
  }
} else {
  stop("Manifest not found: ", manifest_file,
       "\nWithout it there is no time axis and no recording date.")
}

if (file.exists(landmarks_csv)) {
  landmarks <- read.csv(landmarks_csv)
  if ("fish_id" %in% names(landmarks)) {
    landmarks$join_key <- strip_ext(landmarks$image)
    lm_keep <- intersect(c("join_key", "fish_id", "label_source"), names(landmarks))
    dd <- left_join(dd, landmarks[, lm_keep], by = "join_key")
  }
}
if (!"fish_id" %in% names(dd)) dd$fish_id <- NA_character_
if (!"label_source" %in% names(dd)) dd$label_source <- NA_character_

if (!phenotype_column %in% names(dd)) {
  stop("No column called '", phenotype_column, "' - check CIE_conversion.R output.")
}
if (!"seconds" %in% names(dd)) stop("No 'seconds' column - the timing join failed.")
if (!"video" %in% names(dd)) stop("No 'video' column - the manifest join failed.")

no_video <- sum(is.na(dd$video))
if (no_video > 0) {
  warning(no_video, " frame(s) had no matching video in the manifest and were ",
          "left out. Check the manifest and the colour file come from the same run.")
  dd <- dd[!is.na(dd$video), , drop = FALSE]
}

dd$z <- dd[[phenotype_column]]
dd$minutes <- dd$seconds / 60

#=====================================================================#
#  Helpers                                                            #
#=====================================================================#

detect_camera <- function(video) {
  out <- rep("unknown", length(video))
  for (name in names(camera_patterns)) {
    hit <- grepl(camera_patterns[[name]], video, ignore.case = TRUE)
    out[hit & out == "unknown"] <- name
  }
  out
}

first_or_na <- function(frames, column) {
  if (column %in% names(frames) && nrow(frames) > 0) {
    return(as.character(frames[[column]][1]))
  }
  NA_character_
}

# AIC from a residual sum of squares, so linear and exponential fits can
# be compared on the same Dt values. Guarded against a perfect fit.
aic_from_sse <- function(sse, n, n_params) {
  n * log(max(sse, 1e-12) / n) + 2 * n_params
}

#=====================================================================#
#  Fit one video                                                      #
#=====================================================================#

fit_one_video <- function(frames) {
  frames <- frames[order(frames$seconds), , drop = FALSE]
  frames <- frames[!is.na(frames$z) & !is.na(frames$minutes), , drop = FALSE]
  n <- nrow(frames)
  if (n < min_frames) {
    return(list(summary = data.frame(n_frames = n, note = "too few frames",
                                     stringsAsFactors = FALSE),
                per_frame = NULL))
  }

  t <- frames$minutes
  z <- frames$z
  t0 <- t - min(t)                 # time since the first frame
  span <- max(t) - min(t)

  # ---------------- PRIMARY: straight line through raw L* ------------
  trend <- lm(z ~ t)
  coefs <- summary(trend)$coefficients
  slope <- unname(coefs[2, 1])
  slope_se <- unname(coefs[2, 2])
  slope_p <- if (ncol(coefs) >= 4) unname(coefs[2, 4]) else NA_real_
  slope_r2 <- summary(trend)$r.squared
  noise <- sd(residuals(trend))

  # ---------------- SECONDARY: Dt with averaged ends -----------------
  k_start <- max(1, min(z0_start_frames, n - 1))
  k_end <- max(1, min(z_inf_end_frames, n - 1))
  z0 <- mean(head(z, k_start))
  z_inf <- mean(tail(z, k_end))
  capacity_robust <- z_inf - z0

  dt_rate_linear <- NA_real_
  dt_r2_linear <- NA_real_
  aic_linear <- NA_real_
  dt_rate_exp <- NA_real_
  aic_exp <- NA_real_
  half_life <- NA_real_
  Dt <- rep(NA_real_, n)

  if (abs(capacity_robust) > 1e-9) {
    Dt <- (z - z_inf) / (z0 - z_inf)

    # Linear: Dt falls from about 1 towards 0. Reported as a positive
    # speed - the fraction of the response completed per minute.
    dt_line <- lm(Dt ~ t)
    dt_rate_linear <- -unname(coef(dt_line)[2])
    dt_r2_linear <- summary(dt_line)$r.squared
    aic_linear <- aic_from_sse(sum(residuals(dt_line)^2), n, 2)

    # Exponential: Dt = exp(-k * time since first frame). One parameter,
    # fitted on the same scale as the line so the two are comparable.
    # optimize() can't fail to converge the way nls() can.
    sse_exp <- function(k) sum((Dt - exp(-k * t0))^2)
    k_bounds <- c(-1, 20)
    best <- optimize(sse_exp, interval = k_bounds)
    aic_exp <- aic_from_sse(best$objective, n, 1)
    # A k sitting at the edge of the search range is a failed fit, not a
    # real rate - in testing, a fish that went out and came back gave
    # k = 20, which would otherwise be reported as an extremely fast fish.
    if (min(abs(best$minimum - k_bounds)) > 0.01) {
      dt_rate_exp <- best$minimum
      if (dt_rate_exp > 0) half_life <- log(2) / dt_rate_exp
    }
  }

  # The Dt rates divide by (z0 - z_inf). When that difference is no bigger
  # than the measurement noise, Dt is mostly noise divided by noise, and
  # neither Dt rate means anything - flag it rather than report it as real.
  dt_reliable <- is.finite(capacity_robust) &&
    abs(capacity_robust) > reliability_multiple * noise

  # ---------------- ORIGINAL: single most extreme frame --------------
  z0_first <- z[1]
  if (abs(max(z) - z0_first) >= abs(min(z) - z0_first)) {
    z_inf_extreme <- max(z)
  } else {
    z_inf_extreme <- min(z)
  }
  capacity_extreme <- z_inf_extreme - z0_first
  completeness_extreme <- NA_real_
  if (abs(capacity_extreme) > 1e-9) {
    completeness_extreme <- 1 - abs(z[n] - z_inf_extreme) / abs(capacity_extreme)
  }

  summary_row <- data.frame(
    n_frames = n,
    total_minutes = span,
    note = "",

    # primary
    slope_Lstar_per_min = slope,
    slope_se = slope_se,
    slope_p = slope_p,
    slope_r2 = slope_r2,
    trend_direction = if (slope > 0) "lighter" else "darker",
    fitted_change_Lstar = slope * span,
    noise_Lstar = noise,

    # secondary
    z0 = z0,
    z_inf_robust = z_inf,
    capacity_robust = capacity_robust,
    dt_rate_linear = dt_rate_linear,
    dt_r2_linear = dt_r2_linear,
    dt_rate_exponential = dt_rate_exp,
    half_life_min = half_life,
    aic_linear = aic_linear,
    aic_exponential = aic_exp,

    # original
    z_inf_extreme = z_inf_extreme,
    capacity_extreme = capacity_extreme,
    completeness_extreme = completeness_extreme,

    # Is the change big enough, relative to noise, for Dt to mean anything?
    dt_reliable = dt_reliable,

    # Do the endpoint capacities point the same way as the actual trend?
    # Only judged when there IS a trend (p < 0.05) - for a fish with no
    # real trend, the slope's sign is a coin toss and "disagreement" is
    # meaningless, so those are left blank rather than counted.
    extreme_disagrees_with_trend = if (is.finite(slope_p) && slope_p < 0.05)
      as.integer(sign(capacity_extreme) != sign(slope)) else NA_integer_,
    robust_disagrees_with_trend = if (is.finite(slope_p) && slope_p < 0.05)
      as.integer(sign(capacity_robust) != sign(slope)) else NA_integer_,
    stringsAsFactors = FALSE
  )

  per_frame <- data.frame(
    seconds = frames$seconds,
    minutes = t,
    z = z,
    fitted_trend = unname(fitted(trend)),
    Dt_robust = Dt,
    stringsAsFactors = FALSE
  )

  list(summary = summary_row, per_frame = per_frame)
}

#=====================================================================#
#  Run every video                                                    #
#=====================================================================#

videos <- unique(dd$video)
cat("Fitting", length(videos), "video(s)\n\n")

summary_rows <- list()
frame_rows <- list()

for (v in videos) {
  frames <- dd[dd$video == v, , drop = FALSE]
  fitted <- fit_one_video(frames)

  identity <- data.frame(
    video = v,
    fish_id = first_or_na(frames, "fish_id"),
    label_source = first_or_na(frames, "label_source"),
    camera = detect_camera(v),
    background = first_or_na(frames, "background"),
    video_recorded_at = first_or_na(frames, "video_recorded_at"),
    phenotype = phenotype_column,
    stringsAsFactors = FALSE
  )
  summary_rows[[length(summary_rows) + 1]] <- cbind(identity, fitted$summary)

  if (!is.null(fitted$per_frame)) {
    pf <- fitted$per_frame
    pf$video <- v
    pf$fish_id <- identity$fish_id
    pf$camera <- identity$camera
    frame_rows[[length(frame_rows) + 1]] <- pf
  }
}

results <- bind_rows(summary_rows)
if (!"slope_Lstar_per_min" %in% names(results)) {
  stop("No video had at least ", min_frames, " usable frames - nothing to fit. ",
       "Check that the colour file and the manifest come from the same run.")
}

# Recording dates from a camera whose clock was never set.
recorded_day <- suppressWarnings(as.Date(substr(results$video_recorded_at, 1, 10)))
results$recorded_date_valid <- ifelse(is.na(recorded_day), NA,
                                      recorded_day >= earliest_plausible_date)

#=====================================================================#
#  Choose ONE curve shape for every fish                              #
#=====================================================================#

usable <- !is.na(results$slope_Lstar_per_min)
# Only fish whose change is clearly bigger than their noise get a vote on
# the curve shape - a noise-dominated fish fits both shapes badly and
# would just add noise to the choice.
both_fit <- usable & results$dt_reliable %in% TRUE &
  is.finite(results$aic_linear) & is.finite(results$aic_exponential)

total_aic_linear <- sum(results$aic_linear[both_fit])
total_aic_exp <- sum(results$aic_exponential[both_fit])
fish_prefer_exp <- sum(results$aic_exponential[both_fit] < results$aic_linear[both_fit])
fish_prefer_lin <- sum(both_fit) - fish_prefer_exp

if (curve_shape == "auto") {
  chosen_shape <- if (total_aic_exp < total_aic_linear) "exponential" else "linear"
} else {
  chosen_shape <- curve_shape
}

if (chosen_shape == "linear") {
  results$dt_rate <- results$dt_rate_linear
  shape_units <- "fraction of the response completed per minute (linear)"
} else {
  results$dt_rate <- results$dt_rate_exponential
  shape_units <- "k per minute (exponential)"
}
# Fish whose change was no bigger than their noise get no Dt rate in the
# main column, so it can't slip into an analysis by accident. The raw
# per-shape columns still hold the numbers, for checking.
results$dt_rate[!(results$dt_reliable %in% TRUE)] <- NA
results$dt_rate_units <- ifelse(is.na(results$dt_rate), NA, shape_units)
results$curve_shape_used <- ifelse(is.na(results$dt_rate), NA, chosen_shape)

# Tidy column order: who/what, primary, secondary, original, flags.
preferred <- c("video", "fish_id", "label_source", "camera", "background",
               "video_recorded_at", "recorded_date_valid", "phenotype",
               "n_frames", "total_minutes", "note",
               "slope_Lstar_per_min", "slope_se", "slope_p", "slope_r2",
               "trend_direction", "fitted_change_Lstar", "noise_Lstar",
               "z0", "z_inf_robust", "capacity_robust",
               "curve_shape_used", "dt_rate", "dt_rate_units",
               "dt_rate_linear", "dt_r2_linear", "dt_rate_exponential",
               "half_life_min", "aic_linear", "aic_exponential", "dt_reliable",
               "z_inf_extreme", "capacity_extreme", "completeness_extreme",
               "extreme_disagrees_with_trend", "robust_disagrees_with_trend")
ordered <- c(intersect(preferred, names(results)),
             setdiff(names(results), preferred))
results <- results[, ordered]

dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
out_path <- file.path(output_dir, "plasticity_by_video.csv")
write.csv(results, out_path, row.names = FALSE)

if (length(frame_rows) > 0) {
  per_frame <- bind_rows(frame_rows)
  write.csv(per_frame, file.path(output_dir, "plasticity_per_frame.csv"),
            row.names = FALSE)
}

#=====================================================================#
#  Written summary - printed, and saved for your advisor              #
#=====================================================================#

summary_lines <- character(0)
say <- function(...) {
  line <- paste0(...)
  cat(line, "\n")
  summary_lines <<- c(summary_lines, line)
}

ok <- results[usable, , drop = FALSE]
excluded <- results$video[!usable]

say("PLASTICITY SUMMARY - ", phenotype_column, " - ", format(Sys.time(), "%Y-%m-%d %H:%M"))
say("Source: ", lab_file)
say("")
say("Fish analysed: ", nrow(ok), "   excluded (too few frames): ", length(excluded))
if (length(excluded) > 0) say("  excluded: ", paste(excluded, collapse = ", "))
say("")

say("1. PRIMARY - straight-line trend in raw L*")
n_sig <- sum(ok$slope_p < 0.05, na.rm = TRUE)
say("   Fish with a significant trend (p < 0.05): ", n_sig, " of ", nrow(ok),
    "   (expected by chance alone: about ", round(0.05 * nrow(ok), 1), ")")
say("   Got lighter: ", sum(ok$slope_Lstar_per_min > 0),
    "   got darker: ", sum(ok$slope_Lstar_per_min < 0))
say("   Mean slope: ", sprintf("%+.3f", mean(ok$slope_Lstar_per_min)),
    " L*/min   median: ", sprintf("%+.3f", median(ok$slope_Lstar_per_min)))
if (nrow(ok) >= 3) {
  pop_test <- tryCatch(t.test(ok$slope_Lstar_per_min), error = function(e) NULL)
  if (!is.null(pop_test)) {
    say("   Average change different from zero across fish? p = ",
        sprintf("%.3f", pop_test$p.value))
  }
}
say("   Median measurement noise: ", sprintf("%.2f", median(ok$noise_Lstar)), " L*")
say("")

say("2. SECONDARY - protocol Dt, z_inf = mean of last ", z_inf_end_frames,
    " frame(s), z0 = mean of first ", z0_start_frames, " frame(s)")
say("   Curve shape chosen once for all fish: ", toupper(chosen_shape),
    if (curve_shape == "auto") "  (by total AIC)" else "  (set by hand)")
say("   Total AIC - linear: ", sprintf("%.1f", total_aic_linear),
    "   exponential: ", sprintf("%.1f", total_aic_exp))
say("   Fish individually favouring each - linear: ", fish_prefer_lin,
    "   exponential: ", fish_prefer_exp,
    "   (only the ", sum(both_fit), " fish whose change exceeds ",
    reliability_multiple, "x their noise)")
say("   Rate reported as: ", shape_units)
n_unreliable <- sum(!(ok$dt_reliable %in% TRUE))
if (n_unreliable > 0) {
  say("   ", n_unreliable, " fish changed by less than ", reliability_multiple,
      "x their noise - no Dt rate")
  say("   reported for them (dt_rate is blank). Their primary slope still stands.")
}
say("")

say("3. DO THE METHODS AGREE?  (judged only for the ", n_sig,
    " fish with a significant trend)")
ext_bad <- ok$video[ok$extreme_disagrees_with_trend %in% 1]
rob_bad <- ok$video[ok$robust_disagrees_with_trend %in% 1]
say("   Original z_inf (most extreme frame) points AGAINST the trend in: ",
    length(ext_bad), " fish",
    if (length(ext_bad) > 0) paste0("  (", paste(ext_bad, collapse = ", "), ")") else "")
say("   Robust z_inf (end average) points against the trend in: ",
    length(rob_bad), " fish",
    if (length(rob_bad) > 0) paste0("  (", paste(rob_bad, collapse = ", "), ")") else "")
both_cap <- is.finite(ok$fitted_change_Lstar) & is.finite(ok$capacity_robust)
if (sum(both_cap) >= 3) {
  say("   Fitted change vs robust capacity, correlation across fish: r = ",
      sprintf("%.2f", cor(ok$fitted_change_Lstar[both_cap], ok$capacity_robust[both_cap])))
}
low_complete <- ok$video[!is.na(ok$completeness_extreme) & ok$completeness_extreme < 0.5]
if (length(low_complete) > 0) {
  say("   Fish whose most extreme frame was a passing blip (completeness < 0.5): ",
      paste(low_complete, collapse = ", "))
}
say("")

say("4. CAMERAS")
camera_counts <- table(ok$camera)
for (cam in names(camera_counts)) {
  rows <- ok$camera == cam
  say("   ", cam, ": ", camera_counts[[cam]], " fish   mean z0 ",
      sprintf("%.1f", mean(ok$z0[rows])), "   mean slope ",
      sprintf("%+.3f", mean(ok$slope_Lstar_per_min[rows])), " L*/min")
}
big_cams <- names(camera_counts)[camera_counts >= 2]
if (length(big_cams) >= 2) {
  a <- ok$camera == big_cams[1]
  b <- ok$camera == big_cams[2]
  z0_test <- tryCatch(t.test(ok$z0[a], ok$z0[b]), error = function(e) NULL)
  slope_test <- tryCatch(t.test(ok$slope_Lstar_per_min[a], ok$slope_Lstar_per_min[b]),
                         error = function(e) NULL)
  if (!is.null(z0_test)) {
    say("   Baseline L* differs between ", big_cams[1], " and ", big_cams[2],
        "? p = ", sprintf("%.3f", z0_test$p.value))
  }
  if (!is.null(slope_test)) {
    say("   Rate of change differs between them? p = ",
        sprintf("%.3f", slope_test$p.value))
  }
  say("   Colour calibration should remove camera differences. If baseline")
  say("   L* still differs, include camera as a covariate in every model, and")
  say("   check which populations were filmed on which camera - if camera")
  say("   lines up with population, the two cannot be told apart.")
}
bad_dates <- ok$video[ok$recorded_date_valid %in% FALSE]
if (length(bad_dates) > 0) {
  say("   ", length(bad_dates), " video(s) have a recording date before ",
      format(earliest_plausible_date), " - camera clock never set, so those dates")
  say("   are meaningless. Don't use video_recorded_at for them.")
}
say("")
say("Per video: ", out_path)

writeLines(summary_lines, file.path(output_dir, "plasticity_summary.txt"))
cat("Summary saved:", file.path(output_dir, "plasticity_summary.txt"), "\n")
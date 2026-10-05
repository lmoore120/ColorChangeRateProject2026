#---------------------------------------------------------------------#
#                                                                     #
#   Convert the RGB values to CIE L*a*b* colour space                 #
#                                                                     #
#   Based on CIEconverter.R by Marcus Lee.                            #
#   rgb2lab() below is unchanged. What is different:                  #
#     - column positions are worked out FROM THE DATA rather than     #
#       hardcoded. The original had dd[,2:8422] in one place and      #
#       dd[,7:5049] in another, which disagree with each other and    #
#       with the stats script's dd[,8:5056]. Those numbers drifted    #
#       between versions and silently mis-slice a file of a           #
#       different width.                                              #
#     - joins the frame timing from frames_manifest.csv, so you get   #
#       a seconds column for rate-of-change models                    #
#                                                                     #
#---------------------------------------------------------------------#

rm(list = ls())
library(dplyr)

#=====================================================================#
#  SETTINGS                                                           #
#=====================================================================#

colormap_file <- "C:/Users/lamoo/ColorChangeRates2026/files/APHP/white/white_colormaps/colormap_calib.csv"

# Written by extract_frames.py. Gives each frame its true time in seconds,
# which is what a rate-of-change analysis needs.
manifest_file <- "C:/Users/lamoo/ColorChangeRates2026/files/APHP/white/white_frames/frames_manifest.csv"

output_dir <- "C:/Users/lamoo/ColorChangeRates2026/files/APHP/white/white_colormaps/"

# Which background this batch is. Read from the manifest, which takes it
# from the folder name at extraction time - so it can't drift out of step
# with the data the way a setting edited by hand can. Only used as a
# fallback if the manifest has no background column.
background <- "unknown"

# Which background this batch is. Read from the manifest, which takes it
# from the folder name at extraction time - so it can't drift out of step
# with the data the way a setting edited by hand can. Only used as a
# fallback if the manifest has no background column.
background <- "unknown"

#=====================================================================#
#  Read and work out the layout                                        #
#=====================================================================#

dd <- read.csv(colormap_file)
cat("Read", nrow(dd), "rows x", ncol(dd), "columns\n")

# Drop an unnamed row-number column if read.csv added one.
if (names(dd)[1] %in% c("X", "")) {
  dd <- dd[, -1]
}

# Colormesh writes r_<point>, g_<point>, b_<point> AND x_<point>, y_<point>
# for every sampling point - five columns per point, not three. Relying on
# COLUMN POSITION to find the colour block (as the original script did)
# only works if the file happens to list every r/g/b column before every
# x/y column; a different column order - which is exactly what produced
# this run's file - breaks it, and a THIRD order could pass the row-count
# checks while silently pairing the wrong r, g, b values together with no
# error at all. Selecting by NAME and pairing each r_ column with its own
# g_/b_ by matching point name removes the assumption entirely - the
# result is correct regardless of what order Colormesh wrote the columns.
r_cols <- grep("^r_", names(dd), value = TRUE)
if (length(r_cols) == 0) {
  stop("No columns starting with 'r_' found in ", colormap_file,
       " - this doesn't look like a Colormesh output file.")
}
point_names <- sub("^r_", "", r_cols)
g_cols <- paste0("g_", point_names)
b_cols <- paste0("b_", point_names)
missing_g <- g_cols[!g_cols %in% names(dd)]
missing_b <- b_cols[!b_cols %in% names(dd)]
if (length(missing_g) > 0 || length(missing_b) > 0) {
  stop("Found r_ columns with no matching g_/b_ column for the same point, e.g. ",
       paste(head(c(missing_g, missing_b), 5), collapse = ", "),
       " - the colour columns aren't the triplets this expects.")
}

n_triplets <- length(r_cols)
colour_col_names <- c(r_cols, g_cols, b_cols)
metadata_cols <- which(!names(dd) %in% colour_col_names)

cat("Metadata columns:", length(metadata_cols),
    "| colour columns:", length(colour_col_names),
    "(", n_triplets, "points x r,g,b )\n")
cat("Sampling points:", n_triplets,
    "  (published method used 1683)\n")
if (abs(n_triplets - 1683) > 5) {
  warning("Sampling density differs from the published method (1683 points).")
}

# Three matrices, column i of each belonging to the SAME point - built by
# name lookup, so the pairing is correct no matter the original order.
r_matrix <- dd[, r_cols, drop = FALSE]
g_matrix <- dd[, g_cols, drop = FALSE]
b_matrix <- dd[, b_cols, drop = FALSE]

# rgb2lab expects 0-1. Colormesh works in imager's 0-1 range, but rescale
# if this file happens to be 0-255 rather than failing the range check.
# Checked on the COLOUR columns only now - x_/y_ pixel coordinates are
# routinely in the hundreds and would have wrongly tripped this check
# before, when they were still mixed into the same matrix.
rgb_max <- max(r_matrix, g_matrix, b_matrix, na.rm = TRUE)
if (rgb_max > 1) {
  cat("Values look like 0-255; rescaling to 0-1.\n")
  r_matrix <- r_matrix / 255
  g_matrix <- g_matrix / 255
  b_matrix <- b_matrix / 255
}

#=====================================================================#
#  rgb2lab - unchanged from the original script                        #
#=====================================================================#

rgb2lab <- function(r, g, b) {
  if(any(r < 0 | r > 1) || any(g < 0 | g > 1) || any(b < 0 | b > 1)) {
    stop("R, G, B values must be between 0 and 1.")
  }

  gamma_correct <- function(c) {
    ifelse(c <= 0.04045,
           c / 12.92,
           ((c + 0.055) / 1.055)^2.4)
  }

  r_lin <- gamma_correct(r)
  g_lin <- gamma_correct(g)
  b_lin <- gamma_correct(b)

  X <- r_lin * 0.4124564 + g_lin * 0.3575761 + b_lin * 0.1804375
  Y <- r_lin * 0.2126729 + g_lin * 0.7151522 + b_lin * 0.0721750
  Z <- r_lin * 0.0193339 + g_lin * 0.1191920 + b_lin * 0.9503041

  X <- X * 100
  Y <- Y * 100
  Z <- Z * 100

  Xn <- 95.047
  Yn <- 100.000
  Zn <- 108.883

  x <- X / Xn
  y <- Y / Yn
  z <- Z / Zn

  f <- function(t) {
    ifelse(t > 0.008856,
           t^(1/3),
           (7.787 * t) + (16 / 116))
  }

  fx <- f(x)
  fy <- f(y)
  fz <- f(z)

  L <- (116 * fy) - 16
  a <- 500 * (fx - fy)
  b <- 200 * (fy - fz)

  data.frame(L = L, a = a, b = b)
}

#=====================================================================#
#  Convert every sampling point                                        #
#=====================================================================#

lab_list <- vector("list", n_triplets)

for (i in seq_len(n_triplets)) {
  lab <- rgb2lab(r_matrix[[i]], g_matrix[[i]], b_matrix[[i]])
  colnames(lab) <- paste0(c("L_", "a_", "b_"), i)
  lab_list[[i]] <- lab
  if (i %% 200 == 0) cat("  converted", i, "of", n_triplets, "points\n")
}

lab_converted <- do.call(cbind, lab_list)
dd_lab <- cbind(dd[, metadata_cols, drop = FALSE], lab_converted)

#=====================================================================#
#  Whole-body averages                                                 #
#=====================================================================#

# grep("L_") would also match "a_" and "b_" columns in some namings, so
# anchor the pattern to the start of the column name.
L_cols <- grep("^L_", colnames(dd_lab))
a_cols <- grep("^a_", colnames(dd_lab))
b_cols <- grep("^b_", colnames(dd_lab))
cat("Averaging over", length(L_cols), "L /", length(a_cols),
    "a /", length(b_cols), "b columns\n")

L_mean <- rowMeans(dd_lab[, L_cols])
a_mean <- rowMeans(dd_lab[, a_cols])
b_mean <- rowMeans(dd_lab[, b_cols])

uni_dd <- cbind(dd[, metadata_cols, drop = FALSE], L_mean, a_mean, b_mean)

#=====================================================================#
#  Attach the frame timings                                            #
#=====================================================================#

# The published design had two timepoints per fish and used a simple
# difference. A frame series needs the actual elapsed time on every row,
# so the models can fit a slope rather than a before/after contrast.
if (file.exists(manifest_file)) {
  manifest <- read.csv(manifest_file)
  key <- names(uni_dd)[1]
  cat("Joining timings on column '", key, "'\n", sep = "")

  strip_ext <- function(x) sub("\\.[A-Za-z0-9]+$", "", x)
  manifest$join_key <- strip_ext(manifest$image)
  uni_dd$join_key <- strip_ext(as.character(uni_dd[[key]]))

  before <- nrow(uni_dd)
  uni_dd <- left_join(uni_dd,
                      manifest[, intersect(c("join_key", "label", "video",
                                             "frame_number", "seconds", "minutes",
                                             "met_sharpness_threshold",
                                             "background"), names(manifest))],
                      by = "join_key")
  unmatched <- sum(is.na(uni_dd$seconds))
  cat("Joined", before - unmatched, "of", before, "rows\n")
  if (unmatched > 0) {
    warning(unmatched, " rows had no timing match - check the image names ",
            "in the manifest against the colormap file.")
  }
  uni_dd$join_key <- NULL

  # Prefer the manifest's own background column.
  if ("background.y" %in% names(uni_dd)) {
    uni_dd$background <- uni_dd$background.y
    uni_dd$background.y <- NULL
    uni_dd$background.x <- NULL
  } else if (!"background" %in% names(uni_dd)) {
    uni_dd$background <- background
  }
  found <- unique(uni_dd$background[!is.na(uni_dd$background)])
  found <- found[found != ""]
  if (length(found) == 0) {
    warning("No background recorded. Re-extract with the frames under a ",
            "folder called white or black.")
    uni_dd$background <- background
  } else if (length(found) > 1) {
    warning("This batch mixes backgrounds: ", paste(found, collapse = ", "),
            ". White and black trials should be processed separately.")
  } else {
    cat("Background:", found, "(from the frame folder)\n")
  }
} else {
  warning("Manifest not found at ", manifest_file,
          " - no timing columns added, so you cannot fit a rate of change.")
}

#=====================================================================#
#  Save                                                                #
#=====================================================================#

label_for_file <- if (exists("found") && length(found) == 1) found else background
write.csv(dd_lab, file.path(output_dir,
          paste0("LAB_colormap_", label_for_file, ".csv")), row.names = FALSE)
write.csv(uni_dd, file.path(output_dir,
          paste0("LAB_average_", label_for_file, ".csv")), row.names = FALSE)

cat("\nWrote:\n")
cat(" full  :", file.path(output_dir,
    paste0("LAB_colormap_", label_for_file, ".csv")), "\n")
cat(" means :", file.path(output_dir,
    paste0("LAB_average_", label_for_file, ".csv")), "\n")

# A quick look at whether the fish actually darkened over the trial.
if ("seconds" %in% names(uni_dd)) {
  cat("\nMean L* by frame number:\n")
  print(uni_dd %>%
          group_by(frame_number) %>%
          summarise(seconds = mean(seconds),
                    L = mean(L_mean, na.rm = TRUE),
                    n = n(), .groups = "drop"))
}
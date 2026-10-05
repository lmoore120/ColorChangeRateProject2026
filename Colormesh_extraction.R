#---------------------------------------------------------------------#
#                                                                     #
#   Whole body colour extraction - adapted for video frame series     #
#                                                                     #
#   Based on Automated_Colour_Extraction.R by Marcus Lee (08/21/2024) #
#   The Colormesh calls are unchanged. What is different:             #
#     - all paths gathered into one settings block at the top         #
#     - RESUME: already-processed frames are skipped, so a crash or a #
#       closed laptop costs you one frame, not a whole overnight run  #
#     - results written incrementally rather than only at the end     #
#     - consensus shape reused between runs, so batches stay          #
#       comparable                                                    #
#     - can be split across several terminals to run in parallel      #
#     - checks that the sampling map really gives 1683 points         #
#                                                                     #
#   NOTE: this file must contain ONE copy of the script. If it is     #
#   ever pasted in twice, the second copy's settings silently         #
#   override the first and the run fails in a few seconds.            #
#                                                                     #
#---------------------------------------------------------------------#

rm(list = ls())

library(Colormesh)   # Most functions come from this package
library(dplyr)       # Data handling
library(imager)      # Needed by several Colormesh functions
library(abind)       # Binds landmark arrays together
library(geomorph)    # Procrustes analysis
library(Morpho)      # Makes the warped images
library(tools)

#=====================================================================#
#  SETTINGS - the only part you should need to edit                   #
#=====================================================================#

image_dir    <- "C:/Users/lamoo/ColorChangeRates2026/files/APHP/white/white_frames/"
landmark_dir <- "C:/Users/lamoo/ColorChangeRates2026/files/APHP/white/white_landmarks/Landmarks/"
calib_dir    <- "C:/Users/lamoo/ColorChangeRates2026/files/APHP/white/white_landmarks/Landmarks_calib/"
warped_dir   <- "C:/Users/lamoo/ColorChangeRates2026/files/APHP/white/white_warped/"
output_dir   <- "C:/Users/lamoo/ColorChangeRates2026/files/APHP/white/white_colormaps/"
calib_values_file <- "C:/Users/lamoo/ColorChangeRates2026/files/RGB_calib_val.csv"
num_passes   <- 7

# Folder holding the original frames (the PNGs from extract_frames.py).
#image_dir <- "C:/Users/lamoo/ColorChangeRates2026/files/trial_run/frames/"

# Folders written by find_fish_and_landmarks.py.
#landmark_dir <- "C:/Users/lamoo/ColorChangeRates2026/files/trial_run/landmarks/Landmarks/"
#calib_dir <- "C:/Users/lamoo/ColorChangeRates2026/files/trial_run/landmarks/Landmarks_calib/"

# Where unwarped images and results should go. Created if missing.
#warped_dir <- "C:/Users/lamoo/ColorChangeRates2026/files/trial_run/warped/"
#output_dir <- "C:/Users/lamoo/ColorChangeRates2026/files/trial_run/colormaps/"

# The 6 reference patch values. Assigned ONCE - a second assignment
# lower down would silently override this one.
calib_values_file <- "C:/Users/lamoo/ColorChangeRates2026/files/RGB_calib_val.csv"

# How the consensus shape is built.
#   "all"        every frame contributes, exactly as the original script did.
#                Correct when each fish appears once. In a frame series it
#                lets a fish with more usable frames pull the mean shape
#                towards its own body.
#   "one_per_id" one frame per fish contributes to the Procrustes fit, then
#                every frame is warped to that consensus. Preferred for
#                repeated measures of the same individuals.
consensus_mode <- "one_per_id"

# Set TRUE only to deliberately rebuild the consensus shape. Leave FALSE
# for normal batched runs: the consensus must stay identical across every
# batch or the colours aren't comparable between them.
recompute_consensus <- FALSE

# How to work out which fish a frame belongs to, for "one_per_id".
# Frames are named LABEL_VIDEO_fNN_tSSSS.png, so the first two underscore
# fields identify the fish. Change if you rename your files.
id_from_name <- function(x) {
  parts <- strsplit(x, "_")[[1]]
  paste(parts[seq_len(min(2, length(parts)))], collapse = "_")
}

# Passed to tri.surf. This controls how many sampling points you get.
# The published method used 1683 points (1640 interior + 43 perimeter),
# which is what gives 1683 x 3 = 5049 observations per photo.
num_passes <- 4
expected_points <- 1683

# Sampling radius in pixels for rgb.measure.
px_radius <- 2

# Process only part of the list, to split the work across sessions or
# across several terminals. Leave as NA to do everything not already done.
start_index <- NA
end_index   <- NA

# ...or pass them on the command line, which is how you run several copies
# at once without keeping four edited versions of this file:
#     Rscript Colormesh_extraction.R 1 75
#     Rscript Colormesh_extraction.R 76 150
# Command-line values override whatever is set above.
command_args <- commandArgs(trailingOnly = TRUE)
if (length(command_args) >= 2) {
  start_index <- as.integer(command_args[1])
  end_index   <- as.integer(command_args[2])
  cat("Doing frames", start_index, "to", end_index,
      "(from the command line)\n")
}

#=====================================================================#
#  Landmark layout - unchanged from the original script                #
#=====================================================================#

perimeter.map <- c(1, 8:16,
                   2, 17,
                   3, 18:19,
                   4, 20:28,
                   5, 29,
                   6, 30:32,
                   7, 33:43)

sliders1 <- make.sliders(perimeter.map, main.lms = 1:7)

#=====================================================================#
#  Read the landmark files                                             #
#=====================================================================#

dir.create(warped_dir, showWarnings = FALSE, recursive = TRUE)
dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

# Fail early and clearly when a path is wrong, rather than part-way in.
for (needed in c(image_dir, landmark_dir, calib_dir)) {
  if (!dir.exists(needed)) {
    stop("This folder does not exist: ", needed,
         "\nCheck the SETTINGS block at the top of the script.")
  }
}
if (!file.exists(calib_values_file)) {
  stop("Reference values file not found: ", calib_values_file)
}

file_names <- list.files(path = landmark_dir, pattern = "\\_LM\\.TPS$",
                         full.names = TRUE)
if (length(file_names) == 0) {
  stop("No *_LM.TPS files found in ", landmark_dir)
}

# Specimen name = the image file name, which is what the TPS is named after.
extract_specimen_name <- function(file_path) {
  gsub("_LM\\.TPS$", "", basename(file_path))
}
specimen_names <- sapply(file_names, extract_specimen_name, USE.NAMES = FALSE)

cat("Found", length(specimen_names), "landmark files\n")

specimens <- setNames(lapply(file_names, tps2array), specimen_names)
specimen.LM <- abind(specimens, along = 3)
dimnames(specimen.LM)[[3]] <- specimen_names

#---------------------------------------------------------------------#
#  Make every fish face the same way BEFORE the Procrustes fit.
#
#  Procrustes removes position, rotation and size - but not reflection.
#  A fish photographed facing left is the MIRROR IMAGE of one facing
#  right, and gpagen cannot align the two. Averaging them collapses the
#  consensus towards a line and leaves its outline crossing itself, and
#  every frame is then warped onto that tangled target - which is what
#  produced the spiral smearing in the unwarped images.
#
#  So: read the facing from the landmarks (1 is the snout, 4 and 5 are
#  the caudal insertions), and mirror whichever frames point the wrong
#  way. The landmark ORDER is untouched, so landmark 1 is still the
#  snout afterwards.
#---------------------------------------------------------------------#

facing_right <- apply(specimen.LM, 3, function(lm) lm[1, 1] > mean(lm[c(4, 5), 1]))
n_right <- sum(facing_right)
n_left <- sum(!facing_right)
cat("Facing: ", n_right, " right, ", n_left, " left\n", sep = "")

# Mirror the minority, so the fewest frames are altered.
target_facing <- if (n_right >= n_left) TRUE else FALSE
needs_mirror <- facing_right != target_facing

if (any(needs_mirror)) {
  cat("Mirroring", sum(needs_mirror), "frame(s) so every fish faces",
      if (target_facing) "right" else "left", "\n")
  # Reflect about the shape's own vertical midline. Position is removed by
  # Procrustes anyway; what matters is undoing the handedness.
  for (k in which(needs_mirror)) {
    mid <- mean(range(specimen.LM[, 1, k]))
    specimen.LM[, 1, k] <- 2 * mid - specimen.LM[, 1, k]
  }
} else {
  cat("All frames already face the same way.\n")
}

# Same check for dorsal side: a fish lying the other way up is also a
# reflection, and averaging those collapses the shape in the same way.
dorsal_up <- apply(specimen.LM, 3, function(lm)
  mean(lm[c(2, 3), 2]) > mean(lm[c(6, 7), 2]))
if (length(unique(dorsal_up)) > 1) {
  target_dorsal <- sum(dorsal_up) >= sum(!dorsal_up)
  flip_these <- dorsal_up != target_dorsal
  cat("Flipping", sum(flip_these), "frame(s) so the dorsal side is consistent\n")
  for (k in which(flip_these)) {
    mid <- mean(range(specimen.LM[, 2, k]))
    specimen.LM[, 2, k] <- 2 * mid - specimen.LM[, 2, k]
  }
}

# Store which frames were mirrored - the warp needs to mirror the IMAGE
# to match, otherwise the spline is asked to turn the picture inside out.
mirrored_frames <- setNames(needs_mirror, specimen_names)

#=====================================================================#
#  Consensus shape                                                     #
#=====================================================================#

consensus_path <- file.path(output_dir, "consensus_shape.rds")

# When several copies run at once, only the first should build the
# consensus. Two processes writing it together would leave a half-written
# file, and a frame warped to that is silently wrong. Build it once on its
# own (see the note at the bottom), then start the parallel copies.
if (!file.exists(consensus_path) && !is.na(start_index) && start_index > 1) {
  stop("No consensus_shape.rds yet. Run one copy with no arguments first so ",
       "the consensus is built, then start the parallel copies.")
}

if (file.exists(consensus_path) && !recompute_consensus) {
  # CRITICAL for batched processing. Every frame must be warped to the
  # SAME target shape, or colours sampled in different batches aren't
  # comparable. Recomputing the consensus on each run - which is what
  # happens if you just add more landmark files and rerun - silently
  # changes that target. So once a consensus exists it is reused, and
  # you have to opt in to replacing it.
  mean.lm <- readRDS(consensus_path)
  cat("Reusing the existing consensus shape from", consensus_path, "\n")
  cat("Set recompute_consensus <- TRUE to build a new one (this invalidates\n")
  cat("every frame already processed - you would need to delete per_frame/).\n")
} else {
  if (consensus_mode == "one_per_id") {
    fish_id <- sapply(specimen_names, id_from_name, USE.NAMES = FALSE)
    # First frame of each fish, in the order they appear.
    keep <- !duplicated(fish_id)
    cat("Consensus from one frame per fish:", sum(keep), "of",
        length(specimen_names), "frames,", length(unique(fish_id)), "fish\n")
    consensus_input <- specimen.LM[, , keep, drop = FALSE]
  } else {
    cat("Consensus from all", length(specimen_names), "frames\n")
    consensus_input <- specimen.LM
  }

  tmp.reg <- gpagen(consensus_input, curves = sliders1, print.progress = TRUE)
  mean.lm <- tmp.reg$consensus * mean(tmp.reg$Csize)
  saveRDS(mean.lm, consensus_path)
  cat("Consensus shape computed and saved.\n")
}

#=====================================================================#
#  Reference values for the colour standard                            #
#=====================================================================#

known.rgb <- read.csv(calib_values_file)

# The reference file is written 0-255 for readability, but Colormesh works
# in imager's 0-1 range. Rescale if needed rather than assuming either way -
# feeding 0-255 values to a 0-1 calibration silently produces nonsense
# instead of an error.
numeric_cols <- sapply(known.rgb, is.numeric)
if (max(known.rgb[, numeric_cols], na.rm = TRUE) > 1) {
  cat("Reference values look like 0-255; rescaling to 0-1 for Colormesh.\n")
  known.rgb[, numeric_cols] <- known.rgb[, numeric_cols] / 255
}
# rgb.calibrate wants just the R,G,B columns in patch order.
known.rgb <- known.rgb[, numeric_cols, drop = FALSE]
cat("Reference patches:", nrow(known.rgb), "\n")
if (nrow(known.rgb) != 6) {
  warning("Expected 6 reference patches, found ", nrow(known.rgb))
}

#=====================================================================#
#  Which frames still need doing (RESUME)                              #
#=====================================================================#

# Each frame's result is written as its own small CSV. Anything already
# present is skipped, so you can stop and restart freely. This matters at
# 2-5 minutes per frame: a run over thousands of frames WILL be interrupted.
per_frame_dir <- file.path(output_dir, "per_frame")
dir.create(per_frame_dir, showWarnings = FALSE, recursive = TRUE)

done <- gsub("\\.csv$", "", list.files(per_frame_dir, pattern = "\\.csv$"))
todo <- which(!(specimen_names %in% done))

if (!is.na(start_index) && !is.na(end_index)) {
  todo <- todo[todo >= start_index & todo <= end_index]
}

cat(length(done), "already done,", length(todo), "to process\n")
if (length(todo) == 0) {
  cat("Nothing to do. Skipping to the combining step at the bottom.\n")
}

#=====================================================================#
#  Main loop - the Colormesh calls here are unchanged                  #
#=====================================================================#

started <- Sys.time()

for (counter in seq_along(todo)) {
  i <- todo[counter]
  j.names <- specimen_names[i]

  result <- tryCatch({

    tmp.image <- image_reader(image_dir, j.names)
    img.dim <- dim(tmp.image)

    ind.LM <- tps2array(paste0(landmark_dir, j.names, "_LM.TPS"))
    dimnames(ind.LM)[[3]] <- j.names

    # If this frame's shape was mirrored for the consensus, mirror the
    # IMAGE and its landmarks too. Without this the spline is asked to
    # map a left-facing fish onto a right-facing target, which it can
    # only do by folding the picture over itself.
    if (isTRUE(mirrored_frames[[j.names]])) {
      tmp.image <- imager::mirror(tmp.image, "x")
      ind.LM[, 1, ] <- img.dim[1] - ind.LM[, 1, ]
    }

    # TPS files store y from the bottom; imager works from the top.
    orig.lms <- cbind(ind.LM[, 1, ], abs(ind.LM[, 2, ] - img.dim[2]))
    tar.lms  <- cbind(mean.lm[, 1] + img.dim[1] / 2,
                      abs((mean.lm[, 2] + img.dim[2] / 2) - img.dim[2]))

    image_defo <- function(x, y) {
      xs <- c(0:(img.dim[1] - 1))
      ys <- c(0:(img.dim[2] - 1))
      img.long <- as.matrix(expand.grid(xs, ys))
      img.long <- Morpho::tps3d(img.long, tar.lms, orig.lms, threads = 0)
      return(list(x = img.long[, 1], y = img.long[, 2]))
    }
    tmp.warp <- imwarp(tmp.image, map = image_defo, direction = "reverse")

    unwarped_name <- paste0(file_path_sans_ext(j.names), "_unwarped.png")
    save.image(tmp.warp, file = file.path(warped_dir, unwarped_name))

    warped <- list(target = tar.lms, unwarped.names = unwarped_name)
    aligned <- load.image(file.path(warped_dir, unwarped_name))

    specimen.sampling.template <- tri.surf(tri.object = warped$target,
                                           point.map = perimeter.map,
                                           num.passes = num_passes,
                                           corresponding.image = aligned,
                                           flip.delaunay = FALSE)

    uncalib_RGB <- rgb.measure(imagedir = warped_dir,
                               image.names = warped$unwarped.names,
                               delaunay.map = specimen.sampling.template,
                               px.radius = px_radius,
                               linearize.color.space = FALSE)

    calib.LM <- tps2array(paste0(calib_dir, j.names, "_calib_LM.TPS"))
    dimnames(calib.LM)[[3]] <- j.names

    # Calibration reads the ORIGINAL image from disk, which was never
    # mirrored, so the calibration landmarks must not be mirrored either.
    calib_RGB <- rgb.calibrate(uncalib_RGB,
                               imagedir = image_dir,
                               image.names = j.names,
                               calib.file = calib.LM,
                               flip.y.values = FALSE,
                               color.standard.values = known.rgb)

    final.df.calib.1 <- make.colormesh.dataset(df = calib_RGB,
                                               specimen.factors = j.names,
                                               use.perimeter.data = TRUE)

    # Check the sampling density the first time through, while there is
    # still time to change num_passes rather than after 9,600 frames.
    if (counter == 1) {
      # Count by NAME, not by dividing all numeric columns by 3.
      # make.colormesh.dataset writes FIVE columns per sampling point -
      # r_, g_, b_ AND x_, y_ - so dividing the numeric total by 3
      # overstates the density by about 1.67x. That miscount reported
      # 1725 points for a file that actually held 1035.
      column_names <- names(final.df.calib.1)
      n_points <- length(grep("^r_", column_names))
      n_interior <- length(grep("^r_interior", column_names))
      n_perimeter <- length(grep("^r_perimeter", column_names))
      cat("\n--- sampling check ---\n")
      cat("total columns:", length(column_names), "\n")
      cat("sampling points:", n_points,
          sprintf("(%d interior + %d perimeter)", n_interior, n_perimeter), "\n")
      cat("published method used", expected_points, "points\n")
      if (abs(n_points - expected_points) > 50) {
        cat("-> adjust num_passes to get closer, then delete per_frame/*.csv\n")
      }
      if (abs(n_points - expected_points) > 5) {
        warning("Sampling density does not match the published method. ",
                "Adjust num_passes until this is ", expected_points, " points.")
      }
      cat("----------------------\n\n")
    }

    write.csv(final.df.calib.1,
              file.path(per_frame_dir, paste0(j.names, ".csv")),
              row.names = FALSE)
    "ok"

  }, error = function(e) {
    # One bad frame must not end an overnight run. Appended, not
    # overwritten, so every failure in the run is recorded.
    cat("  FAILED:", j.names, "-", conditionMessage(e), "\n")
    cat(paste(Sys.time(), j.names, conditionMessage(e), "\n"),
        file = file.path(output_dir, "failures.log"), append = TRUE)
    "failed"
  })

  elapsed <- as.numeric(difftime(Sys.time(), started, units = "mins"))
  per_frame <- elapsed / counter
  remaining <- (length(todo) - counter) * per_frame
  cat(sprintf("[%d/%d] %s (%s)  %.1f min/frame, ~%.0f min left\n",
              counter, length(todo), j.names, result, per_frame, remaining))
}

#=====================================================================#
#  Combine everything into one file                                    #
#=====================================================================#

all_files <- list.files(per_frame_dir, pattern = "\\.csv$", full.names = TRUE)
cat("\nCombining", length(all_files), "per-frame files...\n")

if (length(all_files) == 0) {
  cat("Nothing to combine yet.\n")
} else {
  final.df.calib <- do.call(rbind, lapply(all_files, read.csv))
  out_path <- file.path(output_dir, "colormap_calib.csv")
  write.csv(final.df.calib, out_path, row.names = FALSE)
  cat("Wrote", nrow(final.df.calib), "rows to", out_path, "\n")
  cat("Next: CIE_conversion.R\n")
}

#=====================================================================#
#  Running several copies at once                                     #
#=====================================================================#
#
# At 2-5 minutes a frame this is the slowest step, and it uses one core.
# Splitting it across terminals is the easiest speed-up available.
#
#   1. FIRST, in one terminal on its own, so the consensus gets built:
#        Rscript Colormesh_extraction.R
#      Let it run until it prints "Consensus shape computed and saved",
#      then stop it with Ctrl+C. (Anything it finished is kept.)
#
#   2. THEN start the copies, one per terminal:
#        Rscript Colormesh_extraction.R 1 75    > log1.txt 2>&1
#        Rscript Colormesh_extraction.R 76 150  > log2.txt 2>&1
#        Rscript Colormesh_extraction.R 151 225 > log3.txt 2>&1
#        Rscript Colormesh_extraction.R 226 300 > log4.txt 2>&1
#
#   3. When they have all finished, run it once more with no arguments to
#      stitch the per-frame files into colormap_calib.csv:
#        Rscript Colormesh_extraction.R
#
# Don't use more copies than your machine has cores - beyond that they
# just take turns and nothing gets faster. Check the count with:
#   echo $env:NUMBER_OF_PROCESSORS
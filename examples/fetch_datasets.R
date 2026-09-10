# fetch_datasets.R
# Downloads the real datasets used in pycrr validation examples.
# Run once from the examples/ directory:
#   Rscript fetch_datasets.R

# mgus2 — monoclonal gammopathy study (Kyle et al. 2002, NEJM)
# Source: R survival package (Therneau, GPL-2)
if (!requireNamespace("survival", quietly = TRUE)) install.packages("survival")
write.csv(survival::mgus2, "mgus2.csv", row.names = FALSE)
cat("Wrote mgus2.csv\n")

# BMT — bone marrow transplant (Klein & Moeschberger 1997)
# Source: R cmprsk package (Gray, GPL-2)
if (!requireNamespace("cmprsk", quietly = TRUE)) install.packages("cmprsk")
write.csv(cmprsk::BMT, "BMT_cmprsk.csv", row.names = FALSE)
cat("Wrote BMT_cmprsk.csv\n")

# transplant — liver transplant waiting list (survival package)
write.csv(survival::transplant, "transplant_numeric.csv", row.names = FALSE)
cat("Wrote transplant_numeric.csv\n")

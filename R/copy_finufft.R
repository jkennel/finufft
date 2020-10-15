# ## This file copies the necessary files from finufft
#
# fn_h <- list.files('inst/finufft', full.names = TRUE, recursive = TRUE, pattern = 'h$')
# fn_h <- fn_h[tools::file_ext(fn_h) != 'sh']
# fn_cpp <- list.files('inst/finufft/include', full.names = TRUE, recursive = TRUE, pattern = 'cpp$')
# 
# fn_c <- list.files('inst/finufft/include', full.names = TRUE, recursive = TRUE, pattern = 'c$')
# 
# 
# fn <- c(fn_h, fn_cpp, fn_c)
# 
# 
# for (i in seq_along(fn)) {
#   tmp <- readLines(fn[i])
#   for (j in seq_along(fn_h)) {
#     tmp <- gsub(paste0("<",basename(fn_h[j]),">"), paste0('"',basename(fn_h[j]),'"'), tmp)
#   }
#   writeLines(tmp, file.path('src', basename(fn[i])))
# }
# 
# 
# 
# fn_cpp <- list.files('inst/finufft/src', full.names = TRUE, recursive = TRUE, pattern = 'cpp$')
# 
# 
# 
# fn <- c(fn_h, fn_cpp)
# 
# 
# for (i in seq_along(fn)) {
#   tmp <- readLines(fn[i])
#   for (j in seq_along(fn_h)) {
#     tmp <- gsub(paste0("<",basename(fn_h[j]),">"), paste0('"',basename(fn_h[j]),'"'), tmp)
#     tmp <- gsub("../contrib/legendre_rule_fast.h", 'legendre_rule_fast.h', tmp)
#   }
#   writeLines(tmp, file.path('src', basename(fn[i])))
# }
# 
# 
# fn_h <- list.files('inst/finufft/contrib', full.names = TRUE, recursive = TRUE, pattern = 'h$')
# 
# fn_cpp <- list.files('inst/finufft/contrib', full.names = TRUE, recursive = TRUE, pattern = 'cpp$')
# 
# fn_c <- list.files('inst/finufft/contrib', full.names = TRUE, recursive = TRUE, pattern = 'c$')
# 
# 
# fn <- c(fn_h, fn_cpp, fn_c)
# 
# # if(!dir.exists('src/contrib')) {
# #   dir.create('src/contrib')
# # }
# 
# for (i in seq_along(fn)) {
#   tmp <- readLines(fn[i])
#   # writeLines(tmp, file.path('src/contrib', basename(fn[i])))
#   writeLines(tmp, file.path('src', basename(fn[i])))
# }
# 
# 
# 
# si <- readLines('src/spreadinterp.cpp')
# pa <- readLines('inst/finufft/src/ker_horner_allw_loop.c')
# pb <- readLines('inst/finufft/src/ker_lowupsampfac_horner_allw_loop.c')
# 
# s <- grep('ker_horner_allw_loop.c', si)
# s2 <- grep('ker_lowupsampfac_horner_allw_loop.c', si)
# n <- length(si)
# 
# si <- c(si[1:(s-1)],
# pa,
# si[(s+1):(s2-1)],
# pb,
# si[(s2+1):n])
# 
# 
# writeLines(si, 'src/spreadinterp.cpp')
# 

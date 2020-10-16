# Placeholder with simple test
n     <- 1e3 * 1e3
n_sub <- 1e5
n1    <- 1e2
n2    <- 1e2
s <- sort(sample(1:n, n_sub))

x_comp <- seq(-3*pi, 3*pi, length.out = sqrt(n))
ex <- expand.grid(x_comp, x_comp)[s,]

cj <- complex(real = rnorm(n_sub), imaginary = rnorm(n_sub))

fk <- matrix(complex(real = rnorm(n), imaginary = rnorm(n)), 
             ncol = sqrt(n), nrow = sqrt(n))

sk <- sort(rnorm(1000))
tk <- sort(rnorm(1000))

# type 1

out_2d1 <- nufft_2d1(xj = ex[,1], 
                     yj = ex[,2], 
                     cj = cj, 
                     n1 = n1, 
                     n2 = n2, 
                     tol = 1e-11, 
                     iflag = -1)
tinytest::expect_equal(dim(out_2d1), c(n1, n2))


# type 2

out_2d2 <- nufft_2d2(xj = ex[,1], 
                     yj = ex[,2], 
                     fk = fk, 
                     tol = 1e-11, 
                     iflag = -1)
tinytest::expect_equal(length(out_2d2), nrow(ex))


# type 3
out_2d3 <- nufft_2d3(xj = ex[,1], 
                     yj = ex[,2],  
                     cj = cj, 
                     sk = sk, 
                     tk = tk, 
                     tol = 1e-11, 
                     iflag = -1)
tinytest::expect_equal(length(out_2d3), length(sk))


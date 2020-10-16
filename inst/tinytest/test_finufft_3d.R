# Placeholder with simple test
n     <- 1e2 * 1e2 * 1e2
n_sub <- 1e5
n1    <- 2e1
n2    <- 2e1
n3    <- 2e1
s <- sort(sample(1:n, n_sub))

x_comp <- seq(-3*pi, 3*pi, length.out = n^(1/3))
ex <- expand.grid(x_comp, x_comp, x_comp)[s,]

cj <- complex(real = rnorm(n_sub), imaginary = rnorm(n_sub))

fk <- array(complex(real = rnorm(n), imaginary = rnorm(n)), 
            dim = c(100,100,100)) 

sk <- sort(rnorm(1000))
tk <- sort(rnorm(1000))
uk <- sort(rnorm(1000))


# type 1

out_3d1 <- nufft_3d1(xj = ex[,1], 
                     yj = ex[,2], 
                     zj = ex[,3], 
                     cj = cj, 
                     n1 = n1, 
                     n2 = n2, 
                     n3 = n3, 
                     tol = 1e-11, 
                     iflag = -1)
tinytest::expect_equal(dim(out_3d1), c(n1, n2, n3))


# type 2

out_3d2 <- nufft_3d2(xj = ex[,1], 
                     yj = ex[,2], 
                     zj = ex[,3], 
                     fk = fk, 
                     tol = 1e-11, 
                     iflag = -1)
tinytest::expect_equal(length(out_3d2), nrow(ex))


# type 3
out_3d3 <- nufft_3d3(xj = ex[,1], 
                     yj = ex[,2], 
                     zj = ex[,3],
                     cj = cj, 
                     sk = sk, 
                     tk = tk, 
                     uk = uk, 
                     tol = 1e-11, 
                     iflag = -1)
tinytest::expect_equal(length(out_3d3), length(sk))


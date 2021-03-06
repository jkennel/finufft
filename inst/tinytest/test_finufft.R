
# Placeholder with simple test
n  <- 1e5
n1 <- 5e4
s <- sort(sample(1:n, n1))

x_comp <- seq(-3*pi, 3*pi, length.out = n)
y_comp <- complex(real = sin(3000 * x_comp) + sin(200 * x_comp) * 0.2, imaginary = 0)

xj <- x_comp[s]
cj <- complex(real = sin(3000 * xj) + sin(200 * xj) * 0.2, imaginary = 0)


# type 1
out_1d1 <- nufft_1d1(xj = xj, cj = cj, n1 = n1, tol = 1e-11, iflag = -1)
tinytest::expect_equal(length(out_1d1), n1)



# # type 2
# fk <- complex(real = rnorm(10000), imaginary = rnorm(10000))
# out_1d2 <- nufft_1d2(xj = xj, fk = fk, tol = 1e-10)
# tinytest::expect_equal(length(out_1d2), length(xj))
# 
# 
# # type 3
# sk <- sort(rnorm(1000))
# out_1d3 <- nufft_1d3(xj = xj, cj = cj, sk = sk, tol = 1e-10)
# tinytest::expect_equal(length(out_1d3), length(sk))


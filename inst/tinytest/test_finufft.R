
# Placeholder with simple test
n <- 1e5
s <- sort(sample(1:n, 10000))

x_comp <- seq(-3*pi,  3*pi, length.out = n)
y_comp <- complex(real = sin(3000 * x_comp) + sin(200 * x_comp)*0.2, imaginary = 0)

x <- x_comp[s]
c <- complex(real = sin(3000 * x) + sin(200 * x)*0.2, imaginary = 0)

nfft_res <- nufft_1(x, c, 10000, 1e-10, 1)
# plot(sqrt(Re(nfft_res * Conj(nfft_res))), type= 'l')

tinytest::expect_equal(length(nfft_res), 10000)

y <- x
c <- complex(real = sin(3000 * x * y) + sin(200 * x* y)*0.2, imaginary = 0)

nfft_res <- nufft_2(x, y, c, 1000,1000, 1e-9, 1)

tinytest::expect_equal(length(nfft_res), 1e6)


z <- y
c <- complex(real = sin(3000 * x * y * z) + sin(200 * x * y * z)*0.2, imaginary = 0)

nfft_res <- nufft_3(x, y, z, c, 200, 200, 200, 1e-9, 1)

tinytest::expect_equal(length(nfft_res), 8e6)


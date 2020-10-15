
# Placeholder with simple test
n <- 1e5
s <- sort(sample(1:n, 10000))

x_comp <- seq(-pi,  pi, length.out = n)
y_comp <- complex(real = sin(3000 * x_comp) + sin(200 * x_comp)*0.2, imaginary = 0)

x <- x_comp[s]
y <- complex(real = sin(3000 * x) + sin(200 * x)*0.2, imaginary = 0)

nfft_res <- nfft(x, y, 10000)
tinytest::expect_equal(length(nfft_res), 10000)



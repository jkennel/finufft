test_that("nfft 1-D works", {
  
  n <- 1e5
  s <- sort(sample(1:n, 10000))
  
  x_comp <- seq(-pi,  pi, length.out = n)
  y_comp <- complex(real = sin(3000 * x_comp) + sin(200 * x_comp)*0.2, imaginary = 0)
  
  x <- x_comp[s]
  y <- complex(real = sin(3000 * x) + sin(200 * x)*0.2, imaginary = 0)
  
  nfft_res <- nfft(x, y, 10000)
  
  fft_res <- fft(y_comp)
  
  plot(seq(1, n, 10), sqrt(Re(nfft_res * Conj(nfft_res)))/length(s),  type = 'l')
  points(sqrt(Re(fft_res * Conj(fft_res)))/(n), type = 'l', col = 'red')
  
})

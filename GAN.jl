# This file contains the functions needed to define the gans
function crossEntTrue(x)
  return max(x, 0) - x + log(1 + exp(-abs(x)))
end

function crossEntFalse(x)
  return max(x, 0) + log(1 + exp(-abs(x)))
end

# creates a random orthogonal matrix for initialization
function rand_orth(m, n)
  @assert(m>=n)
  A = randn(m,n)
  return Matrix(qr(A).Q)
end

function ReLU(x)
  return max(x,zero(x))
end

# This function applies a dense net of specified depth, size, and nonlinearity
# with weights given by the first argument to the input data.
function denseNet(weights, data, inDim, outDim, depth, n_hidden, nonlin)
  offs = 1
  res = nonlin.(reshape(weights[offs : (offs + inDim * n_hidden - 1)], 
                        (n_hidden, inDim)) * data)
  offs += inDim * n_hidden
  for k = 1 : depth
    res = nonlin.(reshape(weights[offs : (offs + n_hidden^2 - 1)], 
                          (n_hidden, n_hidden)) * res)
    offs += n_hidden^2
  end
  return reshape(weights[offs : (offs + outDim * n_hidden - 1)], (outDim, n_hidden)) * res
end

# Creates a vector of suitable size to be the weights of a dense net and
# initializes each layer as a random orthonormal matrix.
function init_denseNet(inDim, outDim, depth, n_hidden, scale=0.8)
  if n_hidden >= inDim
    out = (scale * rand_orth(n_hidden, inDim))[:]
  else
    out = (scale * rand_orth(inDim, n_hidden)')[:]
  end

  for k = 1 : depth
    append!(out, scale * rand_orth(n_hidden, n_hidden)[:])
  end
  append!(out, (scale * rand_orth(n_hidden,outDim)')[:])
  return out
end

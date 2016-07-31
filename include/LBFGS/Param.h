// Copyright (C) 2016 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef PARAM_H
#define PARAM_H

#include <Eigen/Core>
#include <stdexcept>  // std::invalid_argument


namespace LBFGSpp {


template <typename Scalar = double>
class LBFGSParam
{
public:
    ///
    /// The number of corrections to approximate the inverse hessian matrix.
    /// The L-BFGS routine stores the computation results of previous \ref m
    /// iterations to approximate the inverse hessian matrix of the current
    /// iteration. This parameter controls the size of the limited memories
    /// (corrections). The default value is \c 6. Values less than \c 3 are
    /// not recommended. Large values will result in excessive computing time.
    ///
    int    m;
    ///
    /// Tolerance for convergence test.
    /// This parameter determines the accuracy with which the solution is to
    /// be found. A minimization terminates when
    /// \f$||g|| < \epsilon * \max(1, ||x||)\f$,
    /// where ||.|| denotes the Euclidean (L2) norm. The default value is
    /// \c 1e-5.
    ///
    Scalar epsilon;
    ///
    /// The maximum number of iterations.
    /// The optimization process is terminated when the iteration count
    /// exceedes this parameter. Setting this parameter to zero continues an
    /// optimization process until a convergence or error. The default value
    /// is \c 0.
    ///
    int    max_iterations;
    ///
    /// The maximum number of trials for the line search.
    /// This parameter controls the number of function and gradients evaluations
    /// per iteration for the line search routine. The default value is \c 20.
    ///
    int    max_linesearch;
    ///
    /// A parameter to control the accuracy of the line search routine.
    /// The default value is \c 1e-4. This parameter should be greater
    /// than zero and smaller than \c 0.5.
    ///
    Scalar ftol;
    ///
    /// A coefficient for the Wolfe condition.
    /// This parameter is valid only when the backtracking line-search
    /// algorithm is used with the Wolfe condition.
    /// The default value is \c 0.9. This parameter should be greater
    /// the \ref ftol parameter and smaller than \c 1.0.
    ///
    Scalar wolfe;

public:
    LBFGSParam()
    {
        m              = 6;
        epsilon        = Scalar(1e-5);
        max_iterations = 0;
        max_linesearch = 20;
        ftol           = Scalar(1e-4);
        wolfe          = Scalar(0.9);
    }

    inline void check_param() const
    {
        if(m <= 0)
            throw std::invalid_argument("m must be positive");
        if(epsilon <= 0)
            throw std::invalid_argument("epsilon must be positive");
        if(max_iterations < 0)
            throw std::invalid_argument("max_iterations must be non-negative");
        if(max_linesearch <= 0)
            throw std::invalid_argument("max_linesearch must be positive");
        if(ftol <= 0 || ftol >= 0.5)
            throw std::invalid_argument("ftol must be 0 < ftol < 0.5");
        if(wolfe <= ftol || wolfe >= 1)
            throw std::invalid_argument("wolfe must be ftol < wolfe < 1");
    }
};


} // namespace LBFGSpp

#endif // PARAM_H

// Copyright (C) 2016-2023 Yixuan Qiu <yixuan.qiu@cos.name>
// Copyright (C) 2016-2023 Dirk Toewe <DirkToewe@GoogleMail.com>
// Under MIT license

#ifndef LBFGSPP_LINE_SEARCH_NOCEDAL_WRIGHT_H
#define LBFGSPP_LINE_SEARCH_NOCEDAL_WRIGHT_H

#include <Eigen/Core>
#include <stdexcept>

namespace LBFGSpp {

///
/// A line search algorithm for the strong Wolfe condition. Implementation based on:
///
///   "Numerical Optimization" 2nd Edition,
///   Jorge Nocedal and Stephen J. Wright,
///   Chapter 3. Line Search Methods, page 60.
///
template <typename Scalar>
class LineSearchNocedalWright
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

public:
    ///
    /// Line search by Nocedal and Wright (2006).
    ///
    /// \param f        A function object such that `f(x, grad)` returns the
    ///                 objective function value at `x`, and overwrites `grad` with
    ///                 the gradient.
    /// \param param    Parameters for the L-BFGS algorithm.
    /// \param xp       The current point.
    /// \param drt      The current moving direction.
    /// \param step_max The upper bound for the step size that makes x feasible.
    ///                 Can be ignored for the L-BFGS solver.
    /// \param step     In: The initial step length.
    ///                 Out: The calculated step length.
    /// \param fx       In: The objective function value at the current point.
    ///                 Out: The function value at the new point.
    /// \param grad     In: The current gradient vector.
    ///                 Out: The gradient at the new point.
    /// \param dg       In: The inner product between drt and grad.
    ///                 Out: The inner product between drt and the new gradient.
    /// \param x        Out: The new point moved to.
    ///
    template <typename Foo>
    static void LineSearch(Foo& f, const LBFGSParam<Scalar>& param,
                           const Vector& xp, const Vector& drt, const Scalar& step_max,
                           Scalar& step, Scalar& fx, Vector& grad, Scalar& dg, Vector& x)
    {
        // Check the value of step
        if (step <= Scalar(0))
            throw std::invalid_argument("'step' must be positive");

        if (param.linesearch != LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE)
            throw std::invalid_argument("'param.linesearch' must be 'LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE' for LineSearchNocedalWright");

        // To make this implementation more similar to the other line search
        // methods in LBFGSpp, the symbol names from the literature
        // ("Numerical Optimizations") have been changed.
        //
        // Literature | LBFGSpp
        // -----------|--------
        // alpha      | step
        // phi        | fx
        // phi'       | dg

        // The expansion rate of the step size
        const Scalar expansion = Scalar(2);

        // Save the function value at the current x
        const Scalar fx_init = fx;
        // Projection of gradient on the search direction
        const Scalar dg_init = grad.dot(drt);
        // Make sure d points to a descent direction
        if (dg_init > Scalar(0))
            throw std::logic_error("the moving direction increases the objective function value");

        const Scalar test_decr = param.ftol * dg_init,  // Sufficient decrease
            test_curv = -param.wolfe * dg_init;         // Curvature

        // Ends of the line search range (step_lo > step_hi is allowed)
        Scalar step_hi, fx_hi, dg_hi;
        Scalar step_lo = Scalar(0), fx_lo = fx_init, dg_lo = dg_init;

        // STEP 1: Bracketing Phase
        //   Find a range guaranteed to contain a step satisfying strong Wolfe.
        //
        //   See also:
        //     "Numerical Optimization", "Algorithm 3.5 (Line Search Algorithm)".
        int iter = 0;
        for (;;)
        {
            // Evaluate the current step size
            x.noalias() = xp + step * drt;
            fx = f(x, grad);

            iter++;
            if (iter >= param.max_linesearch)
                throw std::runtime_error("the line search routine reached the maximum number of iterations");

            const Scalar dg = grad.dot(drt);

            // Test the sufficient decrease condition
            if (fx - fx_init > step * test_decr || (Scalar(0) < step_lo && fx >= fx_lo))
            {
                step_hi = step;
                fx_hi = fx;
                dg_hi = dg;
                break;
            }
            // If reaching here, then the sufficient decrease condition is satisfied

            // Test the curvature condition
            if (std::abs(dg) <= test_curv)
                return;

            step_hi = step_lo;
            fx_hi = fx_lo;
            dg_hi = dg_lo;
            step_lo = step;
            fx_lo = fx;
            dg_lo = dg;

            if (dg >= Scalar(0))
                break;

            step *= expansion;
        }

        // STEP 2: Zoom Phase
        //   Given a range (step_lo,step_hi) that is guaranteed to
        //   contain a valid strong Wolfe step value, this method
        //   finds such a value.
        //
        //   See also:
        //     "Numerical Optimization", "Algorithm 3.6 (Zoom)".
        for (;;)
        {
            // Use {fx_lo, fx_hi, dg_lo} to make a quadratic interpolation of
            // the function, and the fitted quadratic function is used to
            // estimate the minimum
            //
            // polynomial: p (x) = c0*(x - step)² + c1
            // conditions: p (step_hi) = fx_hi
            //             p (step_lo) = fx_lo
            //             p'(step_lo) = dg_lo

            // We allow fx_hi to be Inf, so first compute a candidate for step size,
            // and test whether NaN occurs
            const Scalar fdiff = fx_hi - fx_lo;
            const Scalar sdiff = step_hi - step_lo;
            const Scalar smid = (step_hi + step_lo) / Scalar(2);
            Scalar step_candid = fdiff * step_lo - smid * sdiff * dg_lo;
            step_candid = step_candid / (fdiff - sdiff * dg_lo);

            // In some cases the interpolation is not a good choice
            // This includes (a) NaN values; (b) too close to the end points; (c) outside the interval
            // In such cases, a bisection search is used
            const bool candid_nan = !(std::isfinite(step_candid));
            const Scalar end_dist = std::min(std::abs(step_candid - step_lo), std::abs(step_candid - step_hi));
            const bool near_end = end_dist < Scalar(0.01) * std::abs(sdiff);
            const bool bisect = candid_nan ||
                (step_candid <= std::min(step_lo, step_hi)) ||
                (step_candid >= std::max(step_lo, step_hi)) ||
                near_end;
            step = bisect ? smid : step_candid;

            // Evaluate the current step size
            x.noalias() = xp + step * drt;
            fx = f(x, grad);

            iter++;
            if (iter >= param.max_linesearch)
                throw std::runtime_error("the line search routine reached the maximum number of iterations");

            const Scalar dg = grad.dot(drt);

            // Test the sufficient decrease condition
            if (fx - fx_init > step * test_decr || fx >= fx_lo)
            {
                if (step == step_hi)
                    throw std::runtime_error("the line search routine failed, possibly due to insufficient numeric precision");

                step_hi = step;
                fx_hi = fx;
                dg_hi = dg;
            }
            else
            {
                // Test the curvature condition
                if (std::abs(dg) <= test_curv)
                    return;

                if (dg * (step_hi - step_lo) >= Scalar(0))
                {
                    step_hi = step_lo;
                    fx_hi = fx_lo;
                    dg_hi = dg_lo;
                }

                if (step == step_lo)
                    throw std::runtime_error("the line search routine failed, possibly due to insufficient numeric precision");

                step_lo = step;
                fx_lo = fx;
                dg_lo = dg;
            }
        }
    }
};

}  // namespace LBFGSpp

#endif  // LBFGSPP_LINE_SEARCH_NOCEDAL_WRIGHT_H

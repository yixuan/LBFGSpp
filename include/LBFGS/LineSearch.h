// Copyright (C) 2016 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LINE_SEARCH_H
#define LINE_SEARCH_H

#include <Eigen/Core>


namespace LBFGSpp {


template <typename Scalar>
class LineSearch
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

public:
    template <typename Foo>
    static void Backtracking(Foo& f, Scalar& fx, Vector& x, Vector& grad,
                             Scalar& step,
                             const Vector& drt, const Vector& xp,
                             const LBFGSParam& param)
    {
        const Scalar dec = 0.5;
        const Scalar inc = 2.1;

        const Scalar fx_init = fx;
        const Scalar dg_init = grad.dot(drt);
        const Scalar dg_test = param.ftol * dg_init;
        Scalar width;

        for(int iter = 0; iter < param.max_linesearch; iter++)
        {
            // x_{k+1} = x_k + step * d_k
            x.noalias() = xp + step * drt;
            // Evaluate this candidate
            fx = f(x, grad);

            if(fx > fx_init + step * dg_test)
            {
                width = dec;
            } else {
                const Scalar dg = grad.dot(drt);
                if(dg < param.wolfe * dg_init)
                {
                    width = inc;
                } else {
                    break;
                }
            }

            step *= width;
        }
    }
};


} // namespace LBFGSpp

#endif // LINE_SEARCH_H

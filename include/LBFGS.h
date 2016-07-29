// Copyright (C) 2016 Yixuan Qiu <yixuan.qiu@cos.name>
// Under MIT license

#ifndef LBFGS_H
#define LBFGS_H

#include <Eigen/Core>
#include "LBFGS/Params.h"
#include "LBFGS/LineSearch.h"


namespace LBFGSpp {


template <typename Scalar>
class LBFGSSolver
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Matrix;
    typedef Eigen::Map<Vector> MapVec;

    const LBFGSParam<Scalar>& m_param;
    Matrix                    m_s;
    Matrix                    m_y;
    Vector                    m_ys;
    Vector                    m_alpha;
    Vector                    m_xp;     // Old x
    Vector                    m_grad;   // Gradient
    Vector                    m_gradp;  // Old gradient
    Vector                    m_drt;    // Moving direction

    inline void reset(int n)
    {
        const int m = m_param.m;
        m_s.resize(n, m);
        m_y.resize(n, m);
        m_ys.resize(m);
        m_alpha.resize(m);
        m_xp.resize(n);
        m_grad.resize(n);
        m_gradp.resize(n);
        m_drt.resize(n);
    }

public:
    LBFGSSolver() :
        m_param(LBFGSParam<Scalar>())
    {
        m_param.check_param();
    }

    LBFGSSolver(const LBFGSParam<Scalar>& param) :
        m_param(param)
    {
        m_param.check_param();
    }

    template <typename Foo>
    void minimize(Foo& f, Vector& x)
    {
        const int n = x.size();
        reset(n);

        // Evaluate function and compute gradient
        double fx = f(x, m_grad);
        double xnorm = x.norm();
        double gnorm = m_grad.norm();

        if(gnorm <= m_param.epsilon * std::max(xnorm, 1.0))
        {
            return;
        }

        // Initial direction
        m_drt.noalias() = -m_grad;
        // Initial step
        double step = 1.0 / m_drt.norm();

        int k = 1;
        int end = 0;
        for(int iter = 0; iter < m_param.max_iterations; iter++)
        {
            // Save the curent x and gradient
            m_xp.noalias() = x;
            m_gradp.noalias() = m_grad;

            LineSearch<Scalar>::Backtracking(f, fx, x, m_grad, step, m_drt, m_xp, m_param);

            // x norm and gradient norm
            xnorm = x.norm();
            gnorm = m_grad.norm();

            if(gnorm <= m_param.epsilon * std::max(xnorm, 1.0))
            {
                return;
            }

            // Update s and y
            MapVec svec(&m_s(0, end), n);
            MapVec yvec(&m_y(0, end), n);
            svec.noalias() = x - m_xp;
            yvec.noalias() = m_grad - m_gradp;

            // ys = y's = 1/rho
            // yy = y'y
            double ys = yvec.dot(svec);
            double yy = yvec.squaredNorm();
            m_ys[end] = ys;

            // Direction = -H * g
            int bound = std::min(m_param.m, k);
            k++;
            end = (end + 1) % m_param.m;

            m_drt.noalias() = -m_grad;

            int j = end;
            for(int i = 0; i < bound; i++)
            {
                j = (j + m_param.m - 1) % m_param.m;
                MapVec sj(&m_s(0, j), n);
                MapVec yj(&m_y(0, j), n);
                m_alpha[j] = sj.dot(m_drt) / m_ys[j];
                m_drt.noalias() -= m_alpha[j] * yj;
            }

            m_drt.noalias() *= (ys / yy);

            for(int i = 0; i < bound; i++)
            {
                MapVec sj(&m_s(0, j), n);
                MapVec yj(&m_y(0, j), n);
                double beta = yj.dot(m_drt) / m_ys[j];
                m_drt.noalias() += (m_alpha[j] - beta) * sj;
                j = (j + 1) % m_param.m;
            }

            step = 1.0;
        }
    }
};


} // namespace LBFGSpp

#endif // LBFGS_H

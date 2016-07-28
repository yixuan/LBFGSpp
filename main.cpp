#include <Eigen/Core>
#include <iostream>

using Eigen::VectorXd;
using Eigen::MatrixXd;
typedef Eigen::Map<Eigen::VectorXd> MapVec;

inline double foo(const VectorXd& x, VectorXd& grad)
{
    const int n = x.size();
    VectorXd d(n);
    for(int i = 0; i < n; i++)
        d[i] = i;

    double f = (x - d).squaredNorm();
    grad.noalias() = 2.0 * (x - d);
    return f;
}

void line_search_backtrack(double& step, double& fx, VectorXd& x, VectorXd& grad,
                           const VectorXd& drt, const VectorXd& xp)
{
    const double dec = 0.5;
    const double inc = 2.1;
    const double ftol = 1e-4;
    const double wolfe = 0.9;
    const int max_linesearch = 20;

    const double fx_init = fx;
    const double dg_init = grad.dot(drt);
    const double dg_test = ftol * dg_init;
    double width;

    std::cout << "  * backtracking: ";

    for(int iter = 0; iter < max_linesearch; iter++)
    {
        // x_{k+1} = x_k + step * d_k
        x.noalias() = xp + step * drt;
        // Evaluate this candidate
        fx = foo(x, grad);

        if(fx > fx_init + step * dg_test)
        {
            width = dec;
            std::cout << "dec ";
        } else {
            const double dg = grad.dot(drt);
            if(dg < wolfe * dg_init)
            {
                width = inc;
                std::cout << "inc ";
            } else {
                break;
            }
        }

        step *= width;
    }

    std::cout << "exit" << std::endl;
}

int main()
{
    const double m_epsilon = 1e-5;
    const int n = 10;
    const int m = 6;
    VectorXd x = VectorXd::Zero(n);

    MatrixXd m_s(n, m);
    MatrixXd m_y(n, m);
    VectorXd m_ys = VectorXd::Zero(m);
    VectorXd m_alpha = VectorXd::Zero(m);

    // Old x
    VectorXd m_xp(n);
    // Gradient
    VectorXd m_grad(n);
    // Old gradient
    VectorXd m_gradp(n);
    // Moving direction
    VectorXd m_drt(n);

    // Evaluate function and compute gradient
    double fx = foo(x, m_grad);
    double xnorm = x.norm();
    double gnorm = m_grad.norm();

    std::cout << "||grad|| = " << gnorm << std::endl;

    if(gnorm <= m_epsilon * std::max(xnorm, 1.0))
    {
        std::cout << "minimum found" << std::endl;
        return 0;
    }

    // Initial direction
    m_drt.noalias() = -m_grad;
    // Initial step
    double step = 1.0 / m_drt.norm();

    int k = 1;
    int end = 0;
    for(int iter = 0; iter < 100; iter++)
    {
        // Save the curent x and gradient
        m_xp.noalias() = x;
        m_gradp.noalias() = m_grad;

        // TODO: update step
        line_search_backtrack(step, fx, x, m_grad, m_drt, m_xp);

        // x norm and gradient norm
        xnorm = x.norm();
        gnorm = m_grad.norm();

        // Report
        std::cout << "||grad|| = " << gnorm << std::endl;

        if(gnorm <= m_epsilon * std::max(xnorm, 1.0))
        {
            std::cout << "minimum found" << std::endl;
            return 0;
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
        int bound = std::min(m, k);
        k++;
        end = (end + 1) % m;

        m_drt.noalias() = -m_grad;

        int j = end;
        for(int i = 0; i < bound; i++)
        {
            j = (j + m - 1) % m;
            MapVec sj(&m_s(0, j), n);
            MapVec yj(&m_y(0, j), n);
            m_alpha[j] = sj.dot(m_drt) / m_ys[j];
            m_drt.noalias() -= m_alpha[j] * yj;
        }

        m_drt *= (ys / yy);

        for(int i = 0; i < bound; i++)
        {
            MapVec sj(&m_s(0, j), n);
            MapVec yj(&m_y(0, j), n);
            double beta = yj.dot(m_drt) / m_ys[j];
            m_drt.noalias() += (m_alpha[j] - beta) * sj;
            j = (j + 1) % m;
        }

        step = 1.0;
    }

    return 0;
}

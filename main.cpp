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
}

int main()
{
    const int n = 10;
    const int m = 5;
    VectorXd x = VectorXd::Zero(n);

    MatrixXd m_s(n, m);
    MatrixXd m_y(n, m);
    VectorXd m_ys(m);
    VectorXd m_alpha(m);

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
    // Initial direction
    m_drt.noalias() = -m_grad;
    // Initial step
    double step = 1.0 / m_drt.norm();

    int k = 1;
    int end = 0;
    double ys, yy;
    for(int iter = 0; iter < 15; iter++)
    {
        // Save the curent x and gradient
        m_xp.noalias() = x;
        m_gradp.noalias() = m_grad;

        // TODO: update step

        // x norm and gradient norm
        xnorm = x.norm();
        gnorm = m_grad.norm();

        // Report
        std::cout << "||grad|| = " << gnorm << std::endl;

        // Update s and y
        MapVec svec(&m_s(0, end), n);
        MapVec yvec(&m_y(0, end), n);
        svec.noalias() = x - m_xp;
        yvec.noalias() = m_grad - m_gradp;

        // ys = y's = 1/rho
        // yy = y'y
        ys = svec.dot(yvec);
        yy = yvec.squaredNorm();
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

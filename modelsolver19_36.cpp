/*
 * 文件名: modelsolver19_36.cpp
 * 文件作用与功能描述:
 * 1. 压裂水平井页岩型复合模型 Group 2 (Model 37-72) 核心计算实现。
 * 2. 实现了页岩气/油藏的特殊介质函数 (f(s) = omega + sqrt(...) * tanh(...))。
 * 3. 【矩阵机制重构】严格按照 MATLAB 的奇点提取分离解析算法重构了 DFN 裂缝系统，淘汰了 n_seg。
 * 4. 【物理法则约束】M12与eta12严格实施物理反比计算，外传入的 eta12 作为无效废弃参数；L 定义修正为全长。
 * 5. 【Laplace严谨耦合】移除了原来外置的时间域井储叠加误差，将 Hegeman 和 Fair 附加压力函数深植入 Laplace 域中解算。
 * 6. 【压敏算法修复】引入泰勒切线平滑外推，彻底解决压敏系数(gamaD)过大导致的曲线杂乱无章、导数崩溃问题。
 */

#include "modelsolver19_36.h"
#include "pressurederivativecalculator.h"

#include <Eigen/Dense>
#include <boost/math/special_functions/bessel.hpp>
#include <boost/math/special_functions/erf.hpp>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <QDebug>
#include <QtConcurrent>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double ModelSolver19_36::safe_bessel_k(int v, double x) {
    if (x < 1e-15) x = 1e-15;
    try { return boost::math::cyl_bessel_k(v, x); } catch (...) { return 0.0; }
}

double ModelSolver19_36::safe_bessel_k_scaled(int v, double x) {
    if (x < 1e-15) x = 1e-15;
    if (x > 600.0) return std::sqrt(M_PI / (2.0 * x));
    try { return boost::math::cyl_bessel_k(v, x) * std::exp(x); } catch (...) { return 0.0; }
}

double ModelSolver19_36::safe_bessel_i_scaled(int v, double x) {
    if (x < 0) x = -x;
    if (x > 600.0) return 1.0 / std::sqrt(2.0 * M_PI * x);
    try { return boost::math::cyl_bessel_i(v, x) * std::exp(-x); } catch (...) { return 0.0; }
}

ModelSolver19_36::ModelSolver19_36(ModelType type)
    : m_type(type), m_highPrecision(true), m_currentN(0) {
    precomputeStehfestCoeffs(12);
}

ModelSolver19_36::~ModelSolver19_36() {}

void ModelSolver19_36::setHighPrecision(bool high) { m_highPrecision = high; }

QString ModelSolver19_36::getModelName(ModelType type, bool verbose)
{
    int id = (int)type + 1;
    QString baseName;
    QString subType;

    if (id <= 12) {
        baseName = QString("页岩型储层试井解释模型%1").arg(id);
        subType = "页岩型+页岩型";
    } else if (id <= 24) {
        baseName = QString("页岩型储层试井解释模型%1").arg(id);
        subType = "页岩型+均质";
    } else {
        baseName = QString("页岩型储层试井解释模型%1").arg(id);
        subType = "页岩型+双重孔隙";
    }

    if (!verbose) return baseName;

    int rem4 = (id - 1) % 4;
    QString strStorage;
    if (rem4 == 0) strStorage = "定井储";
    else if (rem4 == 1) strStorage = "线源解";
    else if (rem4 == 2) strStorage = "Fair模型";
    else strStorage = "Hegeman模型";

    int groupIdx = (id - 1) % 12;
    QString strBoundary;
    if (groupIdx < 4) strBoundary = "无限大外边界";
    else if (groupIdx < 8) strBoundary = "封闭边界";
    else strBoundary = "定压边界";

    return QString("%1\n(%2、%3、%4)").arg(baseName).arg(strStorage).arg(strBoundary).arg(subType);
}

QVector<double> ModelSolver19_36::generateLogTimeSteps(int count, double startExp, double endExp) {
    QVector<double> t;
    if (count <= 0) return t;
    t.reserve(count);
    for (int i = 0; i < count; ++i) {
        double exponent = startExp + (endExp - startExp) * i / (count - 1);
        t.append(pow(10.0, exponent));
    }
    return t;
}

ModelCurveData ModelSolver19_36::calculateTheoreticalCurve(const QMap<QString, double>& params, const QVector<double>& providedTime)
{
    QVector<double> tPoints = providedTime;
    if (tPoints.isEmpty()) tPoints = generateLogTimeSteps(100, -3.0, 4.0);

    double phi = params.value("phi", 0.05);
    double mu = params.value("mu", 0.5);
    double B = params.value("B", 1.2);
    double Ct = params.value("Ct", 5e-4);
    double q = params.value("q", 50.0);
    double h = params.value("h", 20.0);
    double kf = params.value("kf", 50.0); // kf单位 mD

    double L_total = params.value("L", 1000.0);
    double L = L_total / 2.0;

    if (L < 1e-9) L = 500.0;
    if (phi < 1e-12 || mu < 1e-12 || Ct < 1e-12 || kf < 1e-12) {
        return std::make_tuple(tPoints, QVector<double>(tPoints.size(), 0.0), QVector<double>(tPoints.size(), 0.0));
    }

    double td_coeff = 14.4 * kf / (phi * mu * Ct * pow(L, 2.0));
    QVector<double> tD_vec;
    tD_vec.reserve(tPoints.size());
    for(double t : tPoints) tD_vec.append(td_coeff * t);

    QMap<QString, double> calcParams = params;
    int N = (int)calcParams.value("N", 12);
    if (N < 4 || N > 18 || N % 2 != 0) N = 12;
    calcParams["N"] = N;
    precomputeStehfestCoeffs(N);

    if (!calcParams.contains("nf") || calcParams["nf"] < 1) calcParams["nf"] = 9;
    calcParams["L_half"] = L;

    QVector<double> PD_vec, Deriv_vec;
    auto func = std::bind(&ModelSolver19_36::flaplace_composite, this, std::placeholders::_1, std::placeholders::_2);
    calculatePDandDeriv(tD_vec, calcParams, func, PD_vec, Deriv_vec);

    double p_coeff = 1.842e-3 * q * mu * B / (kf * h);
    QVector<double> finalP(tPoints.size()), finalDP(tPoints.size());

    for(int i = 0; i < tPoints.size(); ++i) {
        finalP[i] = p_coeff * PD_vec[i];
    }

    if (tPoints.size() > 2) {
        finalDP = PressureDerivativeCalculator::calculateBourdetDerivative(tPoints, finalP, 0.2);
    } else {
        finalDP.fill(0.0);
    }

    return std::make_tuple(tPoints, finalP, finalDP);
}

void ModelSolver19_36::calculatePDandDeriv(const QVector<double>& tD, const QMap<QString, double>& params,
                                           std::function<double(double, const QMap<QString, double>&)> laplaceFunc,
                                           QVector<double>& outPD, QVector<double>& outDeriv)
{
    int numPoints = tD.size();
    outPD.resize(numPoints);
    outDeriv.resize(numPoints);
    int N = m_currentN;
    double ln2 = 0.6931471805599453;
    double gamaD = params.value("gamaD", 0.02);

    QVector<int> indexes(numPoints);
    std::iota(indexes.begin(), indexes.end(), 0);

    auto calculateSinglePoint = [&](int k) {
        double t = tD[k];
        if (t <= 1e-10) { outPD[k] = 0.0; return; }
        double pd_val = 0.0;
        for (int m = 1; m <= N; ++m) {
            double z = m * ln2 / t;
            double pf = laplaceFunc(z, params);
            if (std::isnan(pf) || std::isinf(pf)) pf = 0.0;
            pd_val += getStehfestCoeff(m, N) * pf;
        }
        double pd_real = pd_val * ln2 / t;

        // --- 摄动法压敏修正 (完美修复曲线震荡版) ---
        if (gamaD > 1e-9) {
            double arg = 1.0 - gamaD * pd_real;
            double arg_min = 1e-3;

            if (arg >= arg_min) {
                pd_real = -1.0 / gamaD * std::log(arg);
            } else {
                // 【核心优化】使用泰勒级数在拐点展开，使得数值被柔性外推，解决导数突变与锯齿
                double val_at_min = -1.0 / gamaD * std::log(arg_min);
                double slope_at_min = -1.0 / (gamaD * arg_min);
                pd_real = val_at_min + slope_at_min * (arg - arg_min);
            }
        }

        if (pd_real <= 1e-15) pd_real = 1e-15;
        outPD[k] = pd_real;
    };
    QtConcurrent::blockingMap(indexes, calculateSinglePoint);
}

double ModelSolver19_36::calc_fs_dual(double u, double omega, double lambda) {
    double one_minus = 1.0 - omega;
    double den = one_minus * u + lambda;
    if (std::abs(den) < 1e-20) return 0.0;
    return (omega * one_minus * u + lambda) / den;
}

double ModelSolver19_36::calc_fs_shale(double u, double omega, double lambda) {
    if (u < 1e-15) return 1.0;
    double one_minus = 1.0 - omega;
    if (one_minus < 1e-9) one_minus = 1e-9;
    if (lambda < 1e-15) lambda = 1e-15;
    double inside_sqrt = 3.0 * one_minus * u / lambda;
    double arg_tanh = std::sqrt(inside_sqrt);
    double front_sqrt = std::sqrt( (lambda * one_minus) / (3.0 * u) );
    return omega + front_sqrt * std::tanh(arg_tanh);
}

double ModelSolver19_36::flaplace_composite(double z, const QMap<QString, double>& p) {
    double M12 = p.value("M12", 5.0);
    double eta12 = 1.0 / M12;

    double L = p.value("L_half", 500.0);
    double Lf = p.value("Lf", 50.0);
    double rm = p.value("rm", 1500.0);
    double re = p.value("re", 20000.0);

    double LfD = (L > 1e-9) ? Lf / L : 0.05;
    double rmD = (L > 1e-9) ? rm / L : 1.25;
    double reD = (L > 1e-9) ? re / L : 25.0;

    int n_fracs = (int)p.value("nf", 9);

    double fs1, fs2;
    double omga1 = p.value("omega1", 0.4);
    double remda1 = p.value("lambda1", 1e-3);

    fs1 = calc_fs_shale(z, omga1, remda1);

    double z_outer = eta12 * z;
    int id = (int)m_type + 1;

    if (id <= 12) {
        double omga2 = p.value("omega2", 0.08);
        double remda2 = p.value("lambda2", 1e-4);
        fs2 = eta12 * calc_fs_shale(z_outer, omga2, remda2);
    } else if (id <= 24) {
        fs2 = eta12;
    } else {
        double omga2 = p.value("omega2", 0.08);
        double remda2 = p.value("lambda2", 1e-4);
        fs2 = eta12 * calc_fs_dual(z_outer, omga2, remda2);
    }

    double pf_base = PWD_composite(z, fs1, fs2, M12, LfD, rmD, reD, n_fracs, m_type);

    int storageType = (int)m_type % 4;
    if (storageType == 1) return pf_base;

    double phi = p.value("phi", 0.05);
    double h = p.value("h", 20.0);
    double Ct = p.value("Ct", 5e-4);
    double C_well = p.value("C", 1e-4);
    double CD = 0.159 * C_well / (phi * h * Ct * L * L);

    if (!p.contains("C") && p.contains("cD")) CD = p.value("cD");

    double S = p.value("S", 1.0);
    if (S < 0.0) S = 0.0;

    double alpha = p.value("alpha", 1e-1);
    double C_phi = p.value("C_phi", 1e-4);

    double P_phiD = 0.0;
    if (storageType == 2) {
        if (std::abs(alpha) > 1e-12) P_phiD = C_phi / (z * (1.0 + alpha * z));
    } else if (storageType == 3) {
        if (std::abs(alpha) > 1e-12) {
            double xx = z * alpha / 2.0;
            double exp_erfc = (xx < 10.0) ? (std::exp(xx * xx) * std::erfc(xx)) : (1.0 / (std::sqrt(M_PI) * xx));
            P_phiD = (C_phi / z) * exp_erfc;
        }
    }

    double pu = z * pf_base + S;
    double pf = 0.0;

    if (storageType == 0) {
        pf = pu / (z * (1.0 + CD * z * pu));
    } else if (storageType == 2 || storageType == 3) {
        double num = pu * (1.0 + CD * z * z * P_phiD);
        double den = z * (1.0 + CD * z * pu);
        if (std::abs(den) > 1e-100) pf = num / den;
        else pf = pf_base;
    }

    return pf;
}

double ModelSolver19_36::PWD_composite(double z, double fs1, double fs2, double M12, double LfD, double rmD, double reD,
                                       int n_fracs, ModelType type) {
    int id = (int)type + 1;
    int groupIdx = (id - 1) % 12;
    bool isInfinite = (groupIdx < 4);
    bool isClosed = (groupIdx >= 4 && groupIdx < 8);
    bool isConstP = (groupIdx >= 8);

    double gama1 = std::sqrt(z * fs1);
    double gama2 = std::sqrt(z * fs2);
    double arg_g1_rm = gama1 * rmD;
    double arg_g2_rm = gama2 * rmD;

    double k0_g1 = safe_bessel_k_scaled(0, arg_g1_rm);
    double k1_g1 = safe_bessel_k_scaled(1, arg_g1_rm);
    double i0_g1 = safe_bessel_i_scaled(0, arg_g1_rm);
    double i1_g1 = safe_bessel_i_scaled(1, arg_g1_rm);

    double k0_g2 = safe_bessel_k_scaled(0, arg_g2_rm);
    double k1_g2 = safe_bessel_k_scaled(1, arg_g2_rm);
    double i0_g2 = safe_bessel_i_scaled(0, arg_g2_rm);
    double i1_g2 = safe_bessel_i_scaled(1, arg_g2_rm);

    double T1_prime = k0_g2;
    double T2_prime = -k1_g2;

    if (!isInfinite && reD > 1e-5) {
        double arg_re = gama2 * reD;
        double k0_re = safe_bessel_k_scaled(0, arg_re);
        double k1_re = safe_bessel_k_scaled(1, arg_re);
        double i0_re = safe_bessel_i_scaled(0, arg_re);
        double i1_re = safe_bessel_i_scaled(1, arg_re);

        double exp_factor = std::exp(2.0 * gama2 * (rmD - reD));

        if (isClosed) {
            double ratio = k1_re / std::max(i1_re, 1e-100);
            T1_prime = ratio * i0_g2 * exp_factor + k0_g2;
            T2_prime = ratio * i1_g2 * exp_factor - k1_g2;
        } else if (isConstP) {
            double ratio = -k0_re / std::max(i0_re, 1e-100);
            T1_prime = ratio * i0_g2 * exp_factor + k0_g2;
            T2_prime = ratio * i1_g2 * exp_factor - k1_g2;
        }
    }

    double Acup_prime   = M12 * gama1 * k1_g1 * T1_prime + gama2 * k0_g1 * T2_prime;
    double Acdown_prime = M12 * gama1 * i1_g1 * T1_prime - gama2 * i0_g1 * T2_prime;
    if (std::abs(Acdown_prime) < 1e-100) Acdown_prime = (Acdown_prime >= 0) ? 1e-100 : -1e-100;
    double Ac_core = Acup_prime / Acdown_prime;

    QVector<double> xwD(n_fracs);
    if (n_fracs == 1) {
        xwD[0] = 0.0;
    } else {
        for (int k = 0; k < n_fracs; ++k) {
            xwD[k] = -0.9 + 1.8 * (double)k / (double)(n_fracs - 1);
        }
    }

    int size = n_fracs + 1;
    Eigen::MatrixXd A_mat(size, size);
    Eigen::VectorXd b_vec(size);
    b_vec.setZero();

    for (int i = 0; i < n_fracs; ++i) {
        for (int j = 0; j < n_fracs; ++j) {
            double dx = std::abs(xwD[i] - xwD[j]);
            double val = 0.0;

            if (i == j) {
                double gL = std::abs(gama1) * LfD;
                double I_singular = 2.0 * LfD * (1.0 - std::log(gL / 2.0));

                auto smooth_part = [&](double a) -> double {
                    if (a < 1e-14) {
                        return -0.57721566490153286 + Ac_core * safe_bessel_i_scaled(0, 0) * std::exp(-2.0 * arg_g1_rm);
                    }
                    double arg_dist = gama1 * a;
                    return safe_bessel_k(0, arg_dist) + std::log(arg_dist / 2.0)
                           + Ac_core * safe_bessel_i_scaled(0, arg_dist) * std::exp(arg_dist - 2.0 * arg_g1_rm);
                };

                double I_smooth = 2.0 * adaptiveGauss(smooth_part, 0.0, LfD, 1e-7, 0, 15);
                val = (I_singular + I_smooth) / (z * 2.0 * LfD);
            } else {
                auto integrand = [&](double a) -> double {
                    double dist_val = std::sqrt(dx * dx + a * a);
                    double arg_dist = gama1 * dist_val;
                    return safe_bessel_k(0, arg_dist) + Ac_core * safe_bessel_i_scaled(0, arg_dist) * std::exp(arg_dist - 2.0 * arg_g1_rm);
                };
                val = adaptiveGauss(integrand, -LfD, LfD, 1e-7, 0, 15) / (z * 2.0 * LfD);
            }
            A_mat(i, j) = z * val;
        }
    }

    for (int i = 0; i < n_fracs; ++i) {
        A_mat(i, n_fracs) = -1.0;
        A_mat(n_fracs, i) = z;
    }
    A_mat(n_fracs, n_fracs) = 0.0;
    b_vec(n_fracs) = 1.0;

    Eigen::VectorXd x_sol = A_mat.fullPivLu().solve(b_vec);
    return x_sol(n_fracs);
}

double ModelSolver19_36::gauss15(std::function<double(double)> f, double a, double b) {
    static const double X[] = { 0.0, 0.20119409, 0.39415135, 0.57097217, 0.72441773, 0.84820658, 0.93729853, 0.98799252 };
    static const double W[] = { 0.20257824, 0.19843149, 0.18616100, 0.16626921, 0.13957068, 0.10715922, 0.07036605, 0.03075324 };
    double h = 0.5 * (b - a);
    double c = 0.5 * (a + b);
    double s = W[0] * f(c);
    for (int i = 1; i < 8; ++i) {
        double dx = h * X[i];
        s += W[i] * (f(c - dx) + f(c + dx));
    }
    return s * h;
}

double ModelSolver19_36::adaptiveGauss(std::function<double(double)> f, double a, double b, double eps, int depth, int maxDepth) {
    double c = (a + b) / 2.0;
    double v1 = gauss15(f, a, b);
    double v2 = gauss15(f, a, c) + gauss15(f, c, b);
    if (depth >= maxDepth || std::abs(v1 - v2) < eps * (std::abs(v2) + 1.0)) return v2;
    return adaptiveGauss(f, a, c, eps/2, depth+1, maxDepth) + adaptiveGauss(f, c, b, eps/2, depth+1, maxDepth);
}

void ModelSolver19_36::precomputeStehfestCoeffs(int N) {
    if (m_currentN == N && !m_stehfestCoeffs.isEmpty()) return;
    m_currentN = N; m_stehfestCoeffs.resize(N + 1);
    for (int i = 1; i <= N; ++i) {
        double s = 0.0;
        int k1 = (i + 1) / 2;
        int k2 = std::min(i, N / 2);
        for (int k = k1; k <= k2; ++k) {
            double num = std::pow((double)k, N / 2.0) * factorial(2 * k);
            double den = factorial(N / 2 - k) * factorial(k) * factorial(k - 1) * factorial(i - k) * factorial(2 * k - i);
            if (den != 0) s += num / den;
        }
        double sign = ((i + N / 2) % 2 == 0) ? 1.0 : -1.0;
        m_stehfestCoeffs[i] = sign * s;
    }
}

double ModelSolver19_36::getStehfestCoeff(int i, int N) {
    if (m_currentN != N) return 0.0;
    if (i < 1 || i > N) return 0.0;
    return m_stehfestCoeffs[i];
}

double ModelSolver19_36::factorial(int n) {
    if(n <= 1) return 1.0;
    double r = 1.0;
    for(int i = 2; i <= n; ++i) r *= i;
    return r;
}


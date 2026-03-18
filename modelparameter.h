/*
 * 文件名: modelparameter.h
 * 文件作用: 项目参数单例类头文件
 * 功能描述:
 * 1. 管理项目核心数据：
 * - 基础物性：孔隙度(phi)、渗透率(k)、有效厚度(h)、粘度(mu)、体积系数(B)、压缩系数(Ct)、产量(q)、井半径(rw)。
 * - 水平井参数：水平井长度(L)、裂缝条数(nf)。
 * - [新增] 变井储模型参数：时间参数(alpha)、压力参数(C_phi)，用于 Fair 和 Hegeman 模型。
 * 2. 负责项目文件(.pwt)的加载与保存，以及关联数据文件(_chart.json, _date.json)的路径管理。
 * 3. 作为全局参数中心，供新建项目、模型计算及恢复默认值使用。
 */

#ifndef MODELPARAMETER_H
#define MODELPARAMETER_H

#include <QString>
#include <QObject>
#include <QJsonObject>
#include <QJsonDocument>
#include <QJsonArray>
#include <QMutex>

class ModelParameter : public QObject
{
    Q_OBJECT

public:
    /**
     * @brief 获取单例实例
     * @return ModelParameter 指针
     */
    static ModelParameter* instance();

    // ========================================================================
    // 项目文件管理
    // ========================================================================

    /**
     * @brief 加载项目文件 (.pwt)
     * @param filePath 文件绝对路径
     * @return 成功返回 true，否则 false
     */
    bool loadProject(const QString& filePath);

    /**
     * @brief 保存当前参数到 .pwt 文件
     * @return 成功返回 true
     */
    bool saveProject();

    /**
     * @brief 关闭项目，清空内存数据并重置为默认值
     */
    void closeProject();

    QString getProjectFilePath() const { return m_projectFilePath; }
    QString getProjectPath() const { return m_projectPath; }
    bool hasLoadedProject() const { return m_hasLoaded; }

    // ========================================================================
    // 基础参数存取 (全局物理参数)
    // ========================================================================

    /**
     * @brief 批量设置主要物理参数 (用于新建项目向导)
     *包含新增的水平井长度 L 和裂缝条数 nf
     */
    void setParameters(double phi, double h, double mu, double B, double Ct,
                       double q, double rw, double L, double nf, const QString& path);

    // --- 变井储参数设置接口 [新增] ---
    void setAlpha(double v);
    void setCPhi(double v);

    // --- 基础参数 Getters ---
    double getPhi() const { return m_phi; }
    double getH() const { return m_h; }
    double getMu() const { return m_mu; }
    double getB() const { return m_B; }
    double getCt() const { return m_Ct; }
    double getQ() const { return m_q; }
    double getRw() const { return m_rw; }

    // --- 水平井参数 Getters ---
    double getL() const { return m_L; }   // 水平井长度
    double getNf() const { return m_nf; } // 裂缝条数

    // --- 变井储参数 Getters [新增] ---
    double getAlpha() const { return m_alpha; } // 变井储时间参数 (h)
    double getCPhi() const { return m_C_phi; }  // 变井储压力参数 (MPa)

    // ========================================================================
    // 拟合结果管理
    // ========================================================================

    // 保存拟合结果到 JSON 对象
    void saveFittingResult(const QJsonObject& fittingData);
    // 获取拟合结果
    QJsonObject getFittingResult() const;

    // ========================================================================
    // 独立数据文件存取 (图表与表格)
    // ========================================================================

    // 保存绘图数据到 "_chart.json"
    void savePlottingData(const QJsonArray& plots);
    QJsonArray getPlottingData() const;

    // 保存表格数据到 "_date.json"
    void saveTableData(const QJsonArray& tableData);
    QJsonArray getTableData() const;

    // 重置所有项目数据（恢复为默认值）
    void resetAllData();

private:
    explicit ModelParameter(QObject* parent = nullptr);
    static ModelParameter* m_instance;

    bool m_hasLoaded;
    QString m_projectPath;
    QString m_projectFilePath;

    // 缓存完整的JSON对象，用于保持未修改数据的完整性
    QJsonObject m_fullProjectData;

    // --- 基础物理参数变量 ---
    double m_phi;   // 孔隙度
    double m_h;     // 有效厚度
    double m_mu;    // 粘度
    double m_B;     // 体积系数
    double m_Ct;    // 综合压缩系数
    double m_q;     // 测试产量
    double m_rw;    // 井筒半径

    // --- 水平井参数 ---
    double m_L;     // 水平井长度
    double m_nf;    // 裂缝条数

    // --- 变井储参数 [新增] ---
    double m_alpha; // 变井储时间参数 (alpha)
    double m_C_phi; // 变井储压力参数 (C_phi)

    // 辅助：获取附属文件的绝对路径
    QString getPlottingDataFilePath() const;
    QString getTableDataFilePath() const;
};

#endif // MODELPARAMETER_H

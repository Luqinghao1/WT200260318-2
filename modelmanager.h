/*
 * 文件名: modelmanager.h
 * 文件作用: 模型管理类头文件
 * 功能描述:
 * 1. 作为核心控制类，管理所有试井模型的生命周期和计算调度。
 * 2. 维护模型列表（共72个），负责实例化对应的 UI 控件和计算求解器。
 * 3. 提供统一的接口 calculateTheoreticalCurve 供外部调用，内部自动分发给 Solver1 或 Solver2。
 * 4. 提供 getDefaultParameters 接口，根据模型类型返回预设的物理参数（含新增的变井储参数）。
 */

#ifndef MODELMANAGER_H
#define MODELMANAGER_H

#include <QObject>
#include <QVector>
#include <QMap>
#include <QStackedWidget>
#include "wt_modelwidget.h"

// 前向声明求解器类
class ModelSolver01_06;
class ModelSolver19_36;

// 定义曲线数据类型
using ModelCurveData = std::tuple<QVector<double>, QVector<double>, QVector<double>>;

class ModelManager : public QObject
{
    Q_OBJECT

public:
    // 全局模型枚举 (Model 1-72)
    // 1-36: Solver1 (夹层/径向复合)
    // 37-72: Solver2 (页岩型)
    enum ModelType {
        // --- Solver 1 ---
        Model_1 = 0, Model_2, Model_3, Model_4, Model_5, Model_6,
        Model_7, Model_8, Model_9, Model_10, Model_11, Model_12,
        Model_13, Model_14, Model_15, Model_16, Model_17, Model_18,
        Model_19, Model_20, Model_21, Model_22, Model_23, Model_24,
        Model_25, Model_26, Model_27, Model_28, Model_29, Model_30,
        Model_31, Model_32, Model_33, Model_34, Model_35, Model_36,

        // --- Solver 2 ---
        Model_37, Model_38, Model_39, Model_40, Model_41, Model_42,
        Model_43, Model_44, Model_45, Model_46, Model_47, Model_48,
        Model_49, Model_50, Model_51, Model_52, Model_53, Model_54,
        Model_55, Model_56, Model_57, Model_58, Model_59, Model_60,
        Model_61, Model_62, Model_63, Model_64, Model_65, Model_66,
        Model_67, Model_68, Model_69, Model_70, Model_71, Model_72
    };

    explicit ModelManager(QWidget* parent = nullptr);
    ~ModelManager();

    /**
     * @brief 初始化模型模块
     * @param parentWidget 父挂载点
     */
    void initializeModels(QWidget* parentWidget);

    /**
     * @brief 切换当前激活的模型
     * @param modelType 模型枚举 ID
     */
    void switchToModel(ModelType modelType);

    /**
     * @brief 获取当前模型类型
     */
    ModelType getCurrentModelType() const { return m_currentModelType; }

    /**
     * @brief 计算理论曲线（核心接口）
     * 自动根据 ID 分发给 Solver1 或 Solver2
     */
    ModelCurveData calculateTheoreticalCurve(ModelType type,
                                             const QMap<QString, double>& params,
                                             const QVector<double>& providedTime = QVector<double>());

    /**
     * @brief 获取模型默认参数
     * 根据模型类型预设合理的参数值（包含 alpha, C_phi 等）
     */
    QMap<QString, double> getDefaultParameters(ModelType type);

    /**
     * @brief 获取模型名称
     */
    static QString getModelTypeName(ModelType type);

    /**
     * @brief 设置高精度模式
     */
    void setHighPrecision(bool high);

    /**
     * @brief 强制更新所有模型的参数显示
     */
    void updateAllModelsBasicParameters();

    // --- 观测数据缓存管理 ---
    void setObservedData(const QVector<double>& t, const QVector<double>& p, const QVector<double>& d);
    void getObservedData(QVector<double>& t, QVector<double>& p, QVector<double>& d) const;
    void clearCache();
    bool hasObservedData() const;

    // 工具函数：生成时间步
    static QVector<double> generateLogTimeSteps(int count, double startExp, double endExp);

signals:
    void modelSwitched(ModelType newModel, ModelType oldModel);
    void calculationCompleted(const QString& title, const QMap<QString, double>& results);

public slots:
    // 响应界面上的"选择模型"按钮
    void onSelectModelClicked();
    // 响应子界面的计算完成信号
    void onWidgetCalculationCompleted(const QString& title, const QMap<QString, double>& results);

private:
    void createMainWidget();
    // 懒加载获取 UI 控件
    WT_ModelWidget* ensureWidget(ModelType type);
    // 懒加载获取求解器
    ModelSolver01_06* ensureSolverGroup1(int index); // index 0-35
    ModelSolver19_36* ensureSolverGroup2(int index); // index 0-35

private:
    QWidget* m_mainWidget;
    QStackedWidget* m_modelStack;

    // 模型列表 (72个)
    QVector<WT_ModelWidget*> m_modelWidgets;

    // 求解器组
    QVector<ModelSolver01_06*> m_solversGroup1; // 对应 Model 1-36
    QVector<ModelSolver19_36*> m_solversGroup2; // 对应 Model 37-72

    ModelType m_currentModelType;

    // 观测数据缓存
    QVector<double> m_cachedObsTime;
    QVector<double> m_cachedObsPressure;
    QVector<double> m_cachedObsDerivative;
};

#endif // MODELMANAGER_H

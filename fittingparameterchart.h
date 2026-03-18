/*
 * 文件名: fittingparameterchart.h
 * 文件作用: 拟合参数图表管理类头文件
 * 修改记录:
 * 1. 新增 generateDefaultParams 静态方法，用于生成默认参数列表。
 * 2. 新增 adjustLimits 静态方法，用于计算参数上下限。
 */

#ifndef FITTINGPARAMETERCHART_H
#define FITTINGPARAMETERCHART_H

#include <QObject>
#include <QTableWidget>
#include <QList>
#include <QMap>
#include <QTimer>
#include "modelmanager.h"

// 拟合参数结构体
struct FitParameter {
    QString name;       // 英文名 (内部标识，如 kf)
    QString displayName;// 中文显示名 (界面显示，如 渗透率)
    double value;       // 当前值
    double min;         // 拟合下限
    double max;         // 拟合上限
    double step;        // 鼠标滚轮调节步长
    bool isFit;         // 是否参与自动拟合 (勾选状态)
    bool isVisible;     // 是否在表格中显示
};

class FittingParameterChart : public QObject
{
    Q_OBJECT
public:
    explicit FittingParameterChart(QTableWidget* parentTable, QObject *parent = nullptr);

    // 设置模型管理器引用
    void setModelManager(ModelManager* m);

    /**
     * @brief 重置为指定模型的默认参数，并自动计算上下限
     * @param type 模型类型
     * @param preserveStates 是否保留之前的"拟合勾选"和"可见性"状态 (默认false)
     */
    void resetParams(ModelManager::ModelType type, bool preserveStates = false);

    // 切换模型（尽可能保留同名参数的数值）
    void switchModel(ModelManager::ModelType newType);

    // 获取当前所有参数列表
    QList<FitParameter> getParameters() const;
    // 设置参数列表并刷新表格
    void setParameters(const QList<FitParameter>& params);

    // 从界面表格读取最新数值更新到内部列表
    void updateParamsFromTable();

    // 根据当前参数值，智能调整上下限和步长 (实例方法)
    void autoAdjustLimits();

    // 静态辅助：获取参数的中文名、单位等显示信息
    static void getParamDisplayInfo(const QString& name, QString& chName, QString& symbol, QString& uniSym, QString& unit);

    // 静态辅助：获取某模型默认应该显示的参数键名列表
    static QStringList getDefaultFitKeys(ModelManager::ModelType type);

    // [新增] 静态辅助：生成指定模型的默认参数列表
    static QList<FitParameter> generateDefaultParams(ModelManager::ModelType type);

    // [新增] 静态辅助：根据参数当前值计算推荐的上下限和步长
    static void adjustLimits(QList<FitParameter>& params);

    // 获取表格中的原始文本（用于支持敏感性分析的逗号分隔输入）
    QMap<QString, QString> getRawParamTexts() const;

signals:
    // 当通过滚轮修改参数时触发，用于实时刷新图表
    void parameterChangedByWheel();

protected:
    // 事件过滤器，用于处理表格上的鼠标滚轮事件
    bool eventFilter(QObject *watched, QEvent *event) override;

private:
    QTableWidget* m_table;          // 关联的UI表格控件
    ModelManager* m_modelManager;   // 模型管理器
    QList<FitParameter> m_params;   // 内部参数数据
    QTimer* m_wheelTimer;           // 滚轮防抖定时器

    // 刷新表格显示
    void refreshParamTable();
    // 向表格添加一行
    void addRowToTable(const FitParameter& p, int& serialNo, bool highlight);

private slots:
    // 表格单元格内容变化槽函数
    void onTableItemChanged(QTableWidgetItem* item);
    // 滚轮操作结束后的延迟处理
    void onWheelDebounceTimeout();
};

#endif // FITTINGPARAMETERCHART_H

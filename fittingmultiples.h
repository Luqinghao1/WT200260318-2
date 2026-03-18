/*
 * 文件名: fittingmultiples.h
 * 文件作用: 多分析对比界面头文件
 * 功能描述:
 * 1. 声明 FittingMultiplesWidget 类，用于展示多个分析的对比结果。
 * 2. 包含一个全屏的 ChartWidget 用于绘制多条曲线。
 * 3. 声明管理统一的悬浮信息窗口 (包含模型、参数、权重三个标签页) 的逻辑。
 * 4. 声明初始化函数，接收多个分析的 JSON 状态数据及曲线选择配置。
 */

#ifndef FITTINGMULTIPLES_H
#define FITTINGMULTIPLES_H

#include <QWidget>
#include <QList>
#include <QJsonObject>
#include <QMap>
#include <QDialog>
#include <QTableWidget>
#include <QTabWidget>
#include "chartwidget.h"
#include "modelmanager.h"
#include "mousezoom.h"
#include "fittingnewdialog.h" // 引入 CurveSelection 结构体定义

namespace Ui {
class FittingMultiplesWidget;
}

class FittingMultiplesWidget : public QWidget
{
    Q_OBJECT

public:
    explicit FittingMultiplesWidget(QWidget *parent = nullptr);
    ~FittingMultiplesWidget();

    // 设置模型管理器 (用于计算理论曲线)
    void setModelManager(ModelManager* m);

    // 初始化：传入多个分析的状态数据 (Name -> JsonObject) 以及 曲线选择配置 (Name -> Selection)
    // 如果 selections 为空，则默认显示所有曲线
    void initialize(const QMap<QString, QJsonObject>& states,
                    const QMap<QString, CurveSelection>& selections = QMap<QString, CurveSelection>());

    // 获取当前状态（用于保存项目）
    QJsonObject getJsonState() const;
    // 加载状态
    void loadState(const QJsonObject& state);

protected:
    // 重写显示/隐藏事件，控制浮动窗口的可见性
    void showEvent(QShowEvent *event) override;
    void hideEvent(QHideEvent *event) override;

private:
    Ui::FittingMultiplesWidget *ui;
    ModelManager* m_modelManager;
    ChartWidget* m_chartWidget;
    MouseZoom* m_plot;

    // 存储各个分析的状态
    QMap<QString, QJsonObject> m_states;

    // [新增] 存储各个分析的曲线显示选项
    QMap<QString, CurveSelection> m_selections;

    // 统一的浮动信息窗口
    QDialog* m_infoDialog;
    QTabWidget* m_infoTabWidget;

    // 浮动窗口内的表格 (分别对应三个标签页)
    QTableWidget* m_tableModelInfo;
    QTableWidget* m_tableParams;
    QTableWidget* m_tableWeights;

    // 初始化绘图组件
    void setupPlot();

    // 初始化统一的悬浮信息窗口
    void initInfoDialog();

    // 刷新图表
    void updateCharts();

    // 刷新浮动窗口内的表格数据
    void updateWindowsData();

    // 辅助：获取不同颜色
    QColor getColor(int index);
};

#endif // FITTINGMULTIPLES_H

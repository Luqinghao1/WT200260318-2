/**
 * @file wt_multidatafittingwidget.h
 * @brief 多数据试井拟合分析核心界面头文件
 *
 * 本代码文件的作用与功能：
 * 1. 声明多数据拟合主界面的整体布局体系、QMdiArea多图表显示系统。
 * 2. 声明数据组的管理结构体 (DataGroup)，用于管理多组测试数据的权重、颜色及显示状态。
 * 3. 声明结果保存、模型拟合计算、抽样设置等交互按钮的槽函数和底层数据处理函数。
 */

#ifndef WT_MULTIDATAFITTINGWIDGET_H
#define WT_MULTIDATAFITTINGWIDGET_H

#include <QWidget>
#include <QMap>
#include <QVector>
#include <QJsonObject>
#include <QMdiArea>
#include <QMdiSubWindow>

#include "modelmanager.h"
#include "fittingparameterchart.h"
#include "fittingchart1.h"
#include "fittingchart2.h"
#include "fittingchart3.h"
#include "mousezoom.h"
#include "fittingcore.h"
#include "fittingchart.h"

namespace Ui {
class WT_MultidataFittingWidget;
}

/**
 * @brief 结构体：用于存储和管理单个实测数据组的核心属性与序列
 */
struct DataGroup {
    QString groupName;              ///< 数据组名称
    double weight;                  ///< 拟合所占权重
    QColor color;                   ///< 图表中分配的颜色
    bool showDeltaP;                ///< 是否显示压差曲线
    bool showDerivative;            ///< 是否显示导数曲线

    QVector<double> time;           ///< 经过抽样或预处理的时间序列
    QVector<double> deltaP;         ///< 经过抽样或预处理的压差序列
    QVector<double> derivative;     ///< 经过抽样或预处理的导数序列

    QVector<double> origTime;       ///< 原始观测时间序列
    QVector<double> origDeltaP;     ///< 原始观测压差序列
    QVector<double> origDerivative; ///< 原始观测导数序列
    QVector<double> rawP;           ///< 原始绝对压力序列
};

class WT_MultidataFittingWidget : public QWidget
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数：初始化组件并绑定UI内部信号槽
     * @param parent 父级窗口指针
     */
    explicit WT_MultidataFittingWidget(QWidget *parent = nullptr);

    /**
     * @brief 析构函数：释放 UI 及其相关内存资源
     */
    ~WT_MultidataFittingWidget();

    /**
     * @brief 设置底层使用的试井模型管理器
     * @param m 模型管理器指针
     */
    void setModelManager(ModelManager* m);

    /**
     * @brief 导入并设置供用户选择的项目观测数据模型
     * @param models 数据名称与标准数据模型的映射表
     */
    void setProjectDataModels(const QMap<QString, QStandardItemModel*>& models);

protected:
    /**
     * @brief 调整窗口大小事件：用于在窗口变化时自动重分布 MDI 绘图区域
     * @param event 尺寸变化事件指针
     */
    void resizeEvent(QResizeEvent* event) override;

    /**
     * @brief 窗口显示事件：用于界面初始展现时进行图表布局刷新
     * @param event 显示事件指针
     */
    void showEvent(QShowEvent* event) override;

private slots:
    /** @brief 槽函数：弹出选择对话框添加新的观测数据组 */
    void on_btnAddData_clicked();
    /** @brief 槽函数：弹出对话框选择试井理论模型 */
    void on_btn_modelSelect_clicked();
    /** @brief 槽函数：弹出参数设置框设定拟合参数上下限及范围 */
    void on_btnSelectParams_clicked();
    /** @brief 槽函数：配置数据平滑和抽样规则 */
    void on_btnSampling_clicked();

    /** @brief 槽函数：启动多数据智能迭代拟合 */
    void on_btnRunFit_clicked();
    /** @brief 槽函数：保存当前多数据拟合的进度状态 */
    void on_btnSave_clicked();
    /** @brief 槽函数：强行中断当前的迭代拟合过程 */
    void on_btnStop_clicked();

    /** @brief 槽函数：强制更新理论模型曲线（常用于手动调整后） */
    void on_btnImportModel_clicked();

    /** @brief 槽函数：处理数据表单元格内容变更（如权重修改） */
    void onDataWeightChanged(int row, int col);
    /** @brief 槽函数：处理数据表点击事件（如点击颜色方块唤出调色板） */
    void onTableCellClicked(int row, int col);
    /** @brief 槽函数：唤出数据管理表格的右键菜单（提供删除等操作） */
    void showContextMenu(const QPoint &pos);
    /** @brief 槽函数：拟合方法/模式切换时触发重新计算 */
    void on_comboFitMode_currentIndexChanged(int index);

    /** @brief 槽函数：响应参数表格中通过鼠标滚轮修改参数数值的事件 */
    void onParameterChangedByWheel();
    /** @brief 槽函数：响应参数表格内文本手动修改完成后的事件 */
    void onParameterTableItemChanged(QTableWidgetItem *item);

    /** @brief 槽函数：接收算法核心传来的每次迭代最新结果并刷新图表 */
    void onIterationUpdate(double err, const QMap<QString,double>& p, const QVector<double>& t, const QVector<double>& p_curve, const QVector<double>& d_curve);
    /** @brief 槽函数：处理拟合结束信号并恢复界面控件状态 */
    void onFitFinished();

    /** @brief 槽函数：将当前的理论及实测曲线数据导出为外部文件 */
    void onExportCurveData();
    /** @brief 槽函数：在半对数图上拖动直线时触发更新地层压力 Pi */
    void onSemiLogLineMoved(double k, double b);

private:
    Ui::WT_MultidataFittingWidget *ui;          ///< UI界面指针
    ModelManager* m_modelManager;               ///< 模型管理器
    FittingCore* m_core;                        ///< 拟合算法核心
    FittingChart* m_chartManager;               ///< 图表管理器

    QMdiArea* m_mdiArea;                        ///< 多文档容器区域
    FittingChart1* m_chartLogLog;               ///< 双对数图表组件
    FittingChart2* m_chartSemiLog;              ///< 半对数图表组件
    FittingChart3* m_chartCartesian;            ///< 标准坐标系图表组件

    QMdiSubWindow* m_subWinLogLog;              ///< 双对数图子窗口
    QMdiSubWindow* m_subWinSemiLog;             ///< 半对数图子窗口
    QMdiSubWindow* m_subWinCartesian;           ///< 标准坐标系图子窗口

    MouseZoom* m_plotLogLog;                    ///< 双对数底层绘图对象
    MouseZoom* m_plotSemiLog;                   ///< 半对数底层绘图对象
    MouseZoom* m_plotCartesian;                 ///< 标准坐标系底层绘图对象

    FittingParameterChart* m_paramChart;        ///< 参数表格管理对象
    QMap<QString, QStandardItemModel*> m_dataMap;///< 项目数据池
    ModelManager::ModelType m_currentModelType; ///< 当前选定的模型类型

    QList<DataGroup> m_dataGroups;              ///< 已经导入的多组测试数据
    bool m_isFitting;                           ///< 标识当前是否正在后台拟合
    bool m_isUpdatingTable;                     ///< 防止表格更新触发死循环的标志位
    double m_userDefinedTimeMax;                ///< 用户自定义的最大拟合时间范围

    /** @brief 初始化图表交互属性（如拖拽、缩放） */
    void setupPlot();
    /** @brief MDI 子窗口布局计算函数，固定图表相对位置 */
    void layoutCharts();
    /** @brief 初始化默认的模型并填充默认参数 */
    void initializeDefaultModel();
    /** @brief 隐藏表格中不需要显示的辅助参数（如 LfD） */
    void hideUnwantedParams();
    /** @brief 从项目级单例中拉取并同步物理基础参数 */
    void loadProjectParams();
    /** @brief 根据数据列表重绘界面的数据管理表格 */
    void refreshDataTable();
    /** @brief 验证多组数据的拟合权重总和是否为 1.0 */
    bool validateDataWeights();
    /** @brief 核心函数：根据参数和数据状态重绘所有的理论模型曲线及散点图 */
    void updateModelCurve(const QMap<QString, double>* explicitParams = nullptr, bool autoScale = false, bool calcError = true);

    /** @brief 工具函数：对数据序列进行一维线性插值 */
    QVector<double> interpolate(const QVector<double>& srcX, const QVector<double>& srcY, const QVector<double>& targetX);
    /** @brief 工具函数：生成覆盖所有数据组范围的对数时间网格 */
    QVector<double> generateCommonTimeGrid();
    /** @brief 工具函数：获取预设的循环调色板颜色 */
    QColor getColor(int index);
};

#endif // WT_MULTIDATAFITTINGWIDGET_H

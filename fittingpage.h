/*
 * 文件名: fittingpage.h
 * 文件作用: 拟合页面容器类头文件
 * 功能描述:
 * 1. 管理多个拟合分析页签，支持 FittingWidget (单分析) 和 FittingMultiplesWidget (多分析对比)。
 * 2. 负责将项目级数据（如模型管理器、观测数据模型集合）传递给各个子页签。
 * 3. 实现多页签的创建、重命名、删除及保存恢复功能。
 * 4. 集成 FittingNewDialog 进行新建分析的交互。
 */

#ifndef FITTINGPAGE_H
#define FITTINGPAGE_H

#include <QWidget>
#include <QJsonObject>
#include <QTabWidget>
#include <QStandardItemModel>
#include <QMap>
#include "modelmanager.h"
#include "fittingmultiples.h" // 包含多分析对比类
#include "fittingnewdialog.h" // [新增] 包含 CurveSelection 结构体定义

// 前置声明
class FittingWidget;

namespace Ui {
class FittingPage;
}

class FittingPage : public QWidget
{
    Q_OBJECT

public:
    explicit FittingPage(QWidget *parent = nullptr);
    ~FittingPage();

    // 设置模型管理器（传递给子页面）
    void setModelManager(ModelManager* m);

    // 设置项目数据模型集合
    void setProjectDataModels(const QMap<QString, QStandardItemModel*>& models);

    // 接收来自外部的数据并设置到当前激活页签 (仅限单分析页签)
    void setObservedDataToCurrent(const QVector<double>& t, const QVector<double>& p, const QVector<double>& d);

    // 初始化/重置基本参数
    void updateBasicParameters();

    // 重置拟合分析
    void resetAnalysis();

    // 从项目文件加载所有拟合分析的状态
    void loadAllFittingStates();

    // 保存所有拟合分析的状态到项目文件
    void saveAllFittingStates();

private slots:
    // 页签管理槽函数
    void on_btnNewAnalysis_clicked();
    void on_btnRenameAnalysis_clicked();
    void on_btnDeleteAnalysis_clicked();

    // 响应子页面的保存请求
    void onChildRequestSave();

private:
    Ui::FittingPage *ui;
    ModelManager* m_modelManager;

    // 存储所有已打开文件的数据模型映射表
    QMap<QString, QStandardItemModel*> m_dataMap;

    // 内部函数：创建新页签 (单分析)
    FittingWidget* createNewTab(const QString& name, const QJsonObject& initData = QJsonObject());

    // 内部函数：创建新页签 (多分析对比)
    // [修改] 增加 selections 参数，用于传递精细化的曲线选择配置
    FittingMultiplesWidget* createNewMultiTab(const QString& name,
                                              const QMap<QString, QJsonObject>& states,
                                              const QMap<QString, CurveSelection>& selections = QMap<QString, CurveSelection>());

    // 生成唯一的页签名称
    QString generateUniqueName(const QString& baseName);

    // 辅助：获取某个页签的 JSON 状态 (兼容两种 Widget)
    QJsonObject getTabState(int index);
};

#endif // FITTINGPAGE_H

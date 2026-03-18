/**
 * @file multidatafittingpage.h
 * @brief 多数据拟合管理页面头文件
 *
 * 本代码文件的作用与功能：
 * 1. 提供多数据分析界面的多标签(Tab)管理功能，允许用户同时打开并切换多个多数据分析任务。
 * 2. 统筹实现“新建分析”、“重命名当前分析页”、“删除当前分析页”等核心界面的交互逻辑。
 * 3. 作为顶层容器，负责接收并向各个独立的多数据拟合子窗口(WT_MultidataFittingWidget)分发项目数据和模型管理器。
 */

#ifndef MULTIDATAFITTINGPAGE_H
#define MULTIDATAFITTINGPAGE_H

#include <QWidget>
#include <QMap>
#include <QStandardItemModel>
#include "modelmanager.h"

namespace Ui {
class multidatafittingpage;
}

class WT_MultidataFittingWidget;

class multidatafittingpage : public QWidget
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数：初始化多数据分析页签管理界面
     * @param parent 父级窗口指针
     */
    explicit multidatafittingpage(QWidget *parent = nullptr);

    /**
     * @brief 析构函数：释放UI及相关资源
     */
    ~multidatafittingpage();

    /**
     * @brief 设置全局模型管理器，并向所有已经打开的子标签页分发
     * @param m 试井模型管理器的指针
     */
    void setModelManager(ModelManager* m);

    /**
     * @brief 设置项目全局数据池，并向所有子标签页同步分发
     * @param models 包含项目观测数据的映射表
     */
    void setProjectDataModels(const QMap<QString, QStandardItemModel*>& models);

private slots:
    /**
     * @brief 槽函数：点击“新建”按钮时触发，创建一个新的多数据分析子窗口及标签
     */
    void on_btnNew_clicked();

    /**
     * @brief 槽函数：点击“重命名”按钮时触发，弹窗提示用户修改当前标签页的名称
     */
    void on_btnRename_clicked();

    /**
     * @brief 槽函数：点击“删除”按钮时触发，关闭并销毁当前处于激活状态的分析标签页
     */
    void on_btnDelete_clicked();

private:
    Ui::multidatafittingpage *ui;                     ///< 界面UI对象指针
    ModelManager* m_modelManager;                     ///< 试井模型管理器指针
    QMap<QString, QStandardItemModel*> m_dataMap;     ///< 项目数据模型映射表

    /**
     * @brief 获取当前处于激活状态的多数据拟合子窗口对象
     * @return 当前子窗口(WT_MultidataFittingWidget)的指针，若无则返回nullptr
     */
    WT_MultidataFittingWidget* currentWidget();
};

#endif // MULTIDATAFITTINGPAGE_H

/*
 * 文件名: paramselectdialog.h
 * 文件作用: 参数选择配置对话框头文件。
 * 功能描述:
 * 1. 声明了参数的显示、选择、限值配置等交互接口。
 * 2. 支持重置默认参数和自动更新限值功能。
 * 3. 包含了表格列重排、高亮行渲染优化（首尾不亮）、且保证高亮不遮挡表格网格线等UI美化相关核心方法的声明。
 */

#ifndef PARAMSELECTDIALOG_H
#define PARAMSELECTDIALOG_H

#include <QDialog>
#include <QTableWidget>
#include <QEvent>
#include "fittingparameterchart.h"

namespace Ui {
class ParamSelectDialog;
}

class ParamSelectDialog : public QDialog
{
    Q_OBJECT

public:
    // 构造函数：初始化参数对话框
    explicit ParamSelectDialog(const QList<FitParameter>& params,
                               ModelManager::ModelType modelType,
                               double fitTime,
                               QWidget *parent = nullptr);
    // 析构函数：释放 UI 资源
    ~ParamSelectDialog();

    // 获取用户修改后的参数列表，供外部提取
    QList<FitParameter> getUpdatedParams() const;

    // 获取用户设定的最大拟合时间
    double getFittingTime() const;

protected:
    // 事件过滤器，用于拦截调节框内的鼠标滚轮事件，防止误修改数值
    bool eventFilter(QObject *obj, QEvent *event) override;

private:
    Ui::ParamSelectDialog *ui;

    // 参数列表副本，用于展示和数据修改绑定
    QList<FitParameter> m_params;

    // 当前选择的模型类型
    ModelManager::ModelType m_modelType;

    // 初始化并渲染表格（包括列顺序重组、背景透明化、高度居中等）
    void initTable();

    // 从界面收集用户填入的最新数据并更新到 m_params
    void collectData();

    // 动态更新指定行外观（淡黄色高亮，且首尾列不亮，防止遮挡边框）
    void updateRowAppearance(int row, bool isFit);

private slots:
    // 确认与取消
    void onConfirm();
    void onCancel();

    // 重置参数与自动适配限值
    void onResetParams();
    void onAutoLimits();
};

#endif // PARAMSELECTDIALOG_H

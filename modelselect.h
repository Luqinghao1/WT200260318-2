/*
 * 文件名: modelselect.h
 * 文件作用: 模型选择对话框头文件
 * 功能描述:
 * 1. 定义模型选择界面的 UI 类。
 * 2. 提供模型选择、参数回显、结果返回的接口。
 * 3. [新增] 支持扩展后的 72 种模型组合的选择逻辑。
 */

#ifndef MODELSELECT_H
#define MODELSELECT_H

#include <QDialog>

namespace Ui {
class ModelSelect;
}

class ModelSelect : public QDialog
{
    Q_OBJECT

public:
    explicit ModelSelect(QWidget *parent = nullptr);
    ~ModelSelect();

    /**
     * @brief 获取选中的模型代码
     * @return 格式如 "modelwidget1" 到 "modelwidget72"
     */
    QString getSelectedModelCode() const;

    /**
     * @brief 获取选中的模型中文名称
     */
    QString getSelectedModelName() const;

    /**
     * @brief 设置当前模型（用于回显）
     * @param code 模型 ID 代码
     */
    void setCurrentModelCode(const QString& code);

private slots:
    void onSelectionChanged();
    void onAccepted();
    void updateInnerOuterOptions();

private:
    void initOptions();

private:
    Ui::ModelSelect *ui;
    QString m_selectedModelCode;
    QString m_selectedModelName;
    bool m_isInitializing;
};

#endif // MODELSELECT_H

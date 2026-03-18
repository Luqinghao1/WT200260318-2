/*
 * 文件名: fittingnewdialog.h
 * 文件作用: 新建分析对话框头文件
 * 功能描述:
 * 1. 提供“空白分析”、“复制单个分析”、“复制多个分析”三种创建模式的选择。
 * 2. [修改] 在多分析复制模式下，提供精细化的内容选择功能（实测压差、实测导数、理论压差、理论导数）。
 * 3. 返回用户设定的新名称、创建模式及源数据列表（包含曲线选择信息）。
 */

#ifndef FITTINGNEWDIALOG_H
#define FITTINGNEWDIALOG_H

#include <QDialog>
#include <QStringList>
#include <QMap>
#include <QTableWidget> // 引入表格控件

namespace Ui {
class FittingNewDialog;
}

// 分析创建模式枚举
enum class AnalysisCreateMode {
    Blank = 0,      // 空白分析
    CopySingle,     // 复制单个
    CopyMultiple    // 复制多个（用于对比）
};

// [新增] 曲线选择结构体，用于记录用户想复制/显示哪些曲线
struct CurveSelection {
    bool showObsP = true;   // 实测压差
    bool showObsD = true;   // 实测导数
    bool showTheoP = true;  // 理论压差
    bool showTheoD = true;  // 理论导数
};

class FittingNewDialog : public QDialog
{
    Q_OBJECT

public:
    // 构造函数：传入当前已有的分析标签页名称列表
    explicit FittingNewDialog(const QStringList& existingNames, QWidget *parent = nullptr);
    ~FittingNewDialog();

    // 获取用户输入的新名称
    QString getNewName() const;

    // 获取用户选择的创建模式
    AnalysisCreateMode getMode() const;

    // 获取用户选择的源分析名称列表 (旧接口，兼容保留)
    QStringList getSourceNames() const;

    // [新增] 获取多分析模式下的详细选择信息 (分析名 -> 选择配置)
    QMap<QString, CurveSelection> getSelectionDetails() const;

private slots:
    // 模式下拉框改变时切换页面
    void onModeChanged(int index);

    // 添加按钮点击槽函数（多选模式）
    void onBtnAddClicked();

    // [修改] 表格右键或双击处理（删除已选）
    void onTableDoubleClicked(const QModelIndex& index);

    // 确认按钮点击槽函数（校验）
    void onAccepted();

private:
    Ui::FittingNewDialog *ui;
    QStringList m_existingNames; // 保存传入的已有分析名称

    // [新增] 用于替代原 listSelected 的表格控件，实现复选框功能
    QTableWidget* m_tableSelected;

    // 辅助函数：初始化表格
    void initSelectionTable();
};

#endif // FITTINGNEWDIALOG_H

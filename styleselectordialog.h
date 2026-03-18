/*
 * 文件名: styleselectordialog.h
 * 文件作用: 通用样式设置对话框头文件
 * 功能描述:
 * 1. 提供统一的界面用于设置颜色、线型、线宽和点形状。
 * 2. 颜色选择改为下拉框模式，内置常用色块。
 * 3. 布局优化：线宽调整至线型下方。
 */

#ifndef STYLESELECTORDIALOG_H
#define STYLESELECTORDIALOG_H

#include <QDialog>
#include <QPen>
#include "qcustomplot.h" // 依赖 QCP 的枚举定义

namespace Ui {
class StyleSelectorDialog;
}

class StyleSelectorDialog : public QDialog
{
    Q_OBJECT

public:
    // 定义显示模式枚举（位标志，支持组合）
    enum Element {
        Color       = 0x01, // 颜色
        Width       = 0x02, // 线宽
        LineStyle   = 0x04, // 线型 (实线/虚线)
        ScatterShape= 0x08, // 散点形状

        // 常用组合预设
        ModeColorOnly = Color,
        ModeLine      = Color | LineStyle | Width, // 线条模式
        ModeScatter   = Color | ScatterShape,      // 散点模式
        ModeAll       = Color | LineStyle | Width | ScatterShape // 全部显示
    };
    Q_DECLARE_FLAGS(Elements, Element)

    /**
     * @brief 构造函数
     * @param elements  指定需要显示的配置项 (例如 ModeLine)
     * @param parent    父窗口
     */
    explicit StyleSelectorDialog(Elements elements = ModeAll, QWidget *parent = nullptr);
    ~StyleSelectorDialog();

    // --- 初始化数据设置 ---
    void setPen(const QPen& pen);
    void setScatterShape(QCPScatterStyle::ScatterShape shape);

    // --- 获取结果数据 ---
    QPen getPen() const;
    QColor getColor() const;
    QCPScatterStyle::ScatterShape getScatterShape() const;

private:
    Ui::StyleSelectorDialog *ui;

    void initUi(Elements elements); // 根据标志位初始化界面
    void initColorComboBox();       // 初始化颜色下拉框
};

Q_DECLARE_OPERATORS_FOR_FLAGS(StyleSelectorDialog::Elements)

#endif // STYLESELECTORDIALOG_H

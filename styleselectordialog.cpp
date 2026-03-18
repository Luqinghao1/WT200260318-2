/*
 * 文件名: styleselectordialog.cpp
 * 文件作用: 通用样式设置对话框实现文件
 * 修改点:
 * 1. 实现了 initColorComboBox，填充常用颜色。
 * 2. 修改布局逻辑，隐藏控件时使用 comboColor 代替 btnColor。
 * 3. 修改了 setPen/getColor 逻辑以适配下拉框。
 */

#include "styleselectordialog.h"
#include "ui_styleselectordialog.h"
#include <QPixmap>
#include <QPainter>

StyleSelectorDialog::StyleSelectorDialog(Elements elements, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::StyleSelectorDialog)
{
    ui->setupUi(this);
    initUi(elements);
}

StyleSelectorDialog::~StyleSelectorDialog()
{
    delete ui;
}

void StyleSelectorDialog::initUi(Elements elements)
{
    // 1. 初始化颜色下拉框 (30种常用色)
    initColorComboBox();

    // 2. 初始化线型下拉框 (中文)
    ui->comboLineStyle->clear();
    ui->comboLineStyle->addItem("实线 (Solid)",       (int)Qt::SolidLine);
    ui->comboLineStyle->addItem("虚线 (Dash)",        (int)Qt::DashLine);
    ui->comboLineStyle->addItem("点线 (Dot)",         (int)Qt::DotLine);
    ui->comboLineStyle->addItem("点划线 (DashDot)",   (int)Qt::DashDotLine);
    ui->comboLineStyle->addItem("双点划线 (DashDotDot)", (int)Qt::DashDotDotLine);
    ui->comboLineStyle->addItem("无 (None)",          (int)Qt::NoPen);

    // 3. 初始化点形状下拉框
    ui->comboShape->clear();
    ui->comboShape->addItem("无 (None)",     (int)QCPScatterStyle::ssNone);
    ui->comboShape->addItem("实心圆 (Disc)", (int)QCPScatterStyle::ssDisc);
    ui->comboShape->addItem("空心圆 (Circle)", (int)QCPScatterStyle::ssCircle);
    ui->comboShape->addItem("正方形 (Square)", (int)QCPScatterStyle::ssSquare);
    ui->comboShape->addItem("三角形 (Triangle)", (int)QCPScatterStyle::ssTriangle);
    ui->comboShape->addItem("菱形 (Diamond)",  (int)QCPScatterStyle::ssDiamond);
    ui->comboShape->addItem("十字 (Cross)",    (int)QCPScatterStyle::ssCross);
    ui->comboShape->addItem("加号 (Plus)",     (int)QCPScatterStyle::ssPlus);
    ui->comboShape->addItem("星形 (Star)",     (int)QCPScatterStyle::ssStar);

    // 4. 根据传入的 elements 标志位，隐藏不需要的控件

    // 颜色设置
    bool showColor = elements.testFlag(Color);
    ui->label_Color->setVisible(showColor);
    ui->comboColor->setVisible(showColor);

    // 线型设置 (注意：线型现在排在第2行)
    bool showLineStyle = elements.testFlag(LineStyle);
    ui->label_LineStyle->setVisible(showLineStyle);
    ui->comboLineStyle->setVisible(showLineStyle);

    // 线宽设置 (注意：线宽现在排在第3行)
    bool showWidth = elements.testFlag(Width);
    ui->label_Width->setVisible(showWidth);
    ui->spinWidth->setVisible(showWidth);

    // 点形状设置
    bool showShape = elements.testFlag(ScatterShape);
    ui->label_Shape->setVisible(showShape);
    ui->comboShape->setVisible(showShape);

    // 5. 自动调整窗口大小
    layout()->setSizeConstraint(QLayout::SetFixedSize);
    adjustSize();
}

void StyleSelectorDialog::initColorComboBox()
{
    ui->comboColor->clear();

    // 定义常用颜色列表 (名称, 颜色值)
    struct ColorItem { QString name; QColor color; };
    QList<ColorItem> colors = {
        {"黑色 (Black)", Qt::black},
        {"红色 (Red)", Qt::red},
        {"蓝色 (Blue)", Qt::blue},
        {"绿色 (Green)", Qt::green},
        {"青色 (Cyan)", Qt::cyan},
        {"品红 (Magenta)", Qt::magenta},
        {"黄色 (Yellow)", Qt::yellow},
        {"深红 (Dark Red)", Qt::darkRed},
        {"深绿 (Dark Green)", Qt::darkGreen},
        {"深蓝 (Dark Blue)", Qt::darkBlue},
        {"深青 (Dark Cyan)", Qt::darkCyan},
        {"深品红 (Dark Magenta)", Qt::darkMagenta},
        {"深黄 (Dark Yellow)", Qt::darkYellow},
        {"灰色 (Gray)", Qt::gray},
        {"深灰 (Dark Gray)", Qt::darkGray},
        {"浅灰 (Light Gray)", Qt::lightGray},
        {"白色 (White)", Qt::white},
        // 自定义颜色
        {"橙色 (Orange)", QColor(255, 165, 0)},
        {"紫色 (Purple)", QColor(128, 0, 128)},
        {"棕色 (Brown)", QColor(165, 42, 42)},
        {"粉色 (Pink)", QColor(255, 192, 203)},
        {"金黄色 (Gold)", QColor(255, 215, 0)},
        {"天蓝 (Sky Blue)", QColor(135, 206, 235)},
        {"蓝绿色 (Teal)", QColor(0, 128, 128)},
        {"海军蓝 (Navy)", QColor(0, 0, 128)},
        {"酸橙绿 (Lime)", QColor(0, 255, 0)},
        {"栗色 (Maroon)", QColor(128, 0, 0)},
        {"紫罗兰 (Violet)", QColor(238, 130, 238)},
        {"珊瑚色 (Coral)", QColor(255, 127, 80)},
        {"靛青 (Indigo)", QColor(75, 0, 130)}
    };

    // 生成带颜色块的图标
    for (const auto& item : colors) {
        QPixmap pix(16, 16);
        pix.fill(item.color);
        // 给色块加个黑边框，防止白色看不见
        QPainter painter(&pix);
        painter.setPen(Qt::gray);
        painter.drawRect(0, 0, 15, 15);

        ui->comboColor->addItem(QIcon(pix), item.name, item.color);
    }
}

void StyleSelectorDialog::setPen(const QPen& pen)
{
    QColor c = pen.color();

    // 在下拉框中查找颜色
    int colorIdx = -1;
    for(int i=0; i<ui->comboColor->count(); ++i) {
        if (ui->comboColor->itemData(i).value<QColor>() == c) {
            colorIdx = i;
            break;
        }
    }

    // 如果是列表里没有的自定义颜色，临时添加到第一项
    if (colorIdx == -1) {
        QPixmap pix(16, 16);
        pix.fill(c);
        QPainter painter(&pix);
        painter.setPen(Qt::gray);
        painter.drawRect(0, 0, 15, 15);

        ui->comboColor->insertItem(0, QIcon(pix), QString("当前颜色 (%1)").arg(c.name()), c);
        ui->comboColor->setCurrentIndex(0);
    } else {
        ui->comboColor->setCurrentIndex(colorIdx);
    }

    // 设置线宽
    ui->spinWidth->setValue(pen.width());

    // 设置线型
    int styleIdx = ui->comboLineStyle->findData((int)pen.style());
    if(styleIdx != -1) ui->comboLineStyle->setCurrentIndex(styleIdx);
    else ui->comboLineStyle->setCurrentIndex(0); // 默认实线
}

void StyleSelectorDialog::setScatterShape(QCPScatterStyle::ScatterShape shape)
{
    int idx = ui->comboShape->findData((int)shape);
    if(idx != -1) ui->comboShape->setCurrentIndex(idx);
    else ui->comboShape->setCurrentIndex(0);
}

QPen StyleSelectorDialog::getPen() const
{
    QPen pen;
    // 从下拉框获取颜色数据
    pen.setColor(ui->comboColor->currentData().value<QColor>());
    pen.setWidth(ui->spinWidth->value());
    pen.setStyle((Qt::PenStyle)ui->comboLineStyle->currentData().toInt());
    return pen;
}

QColor StyleSelectorDialog::getColor() const
{
    return ui->comboColor->currentData().value<QColor>();
}

QCPScatterStyle::ScatterShape StyleSelectorDialog::getScatterShape() const
{
    return (QCPScatterStyle::ScatterShape)ui->comboShape->currentData().toInt();
}

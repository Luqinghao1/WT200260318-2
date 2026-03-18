#include "monitostatew.h"
#include "ui_monitostatew.h"
#include <QMouseEvent>
#include <QDebug>

MonitoStateW::MonitoStateW(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MonitoStateW),
    m_isPressed(false)
{
    ui->setupUi(this);

    // 设置光标为手型，表示可点击
    setCursor(Qt::PointingHandCursor);

    // 调试信息
    qDebug() << "MonitoStateW 构造函数调用";
}

MonitoStateW::~MonitoStateW()
{
    delete ui;
}

void MonitoStateW::setTextInfo(const QString& centerPicStyle, const QString& topPicStyle,
                               const QString& topName, const QString& bottomName)
{
    qDebug() << "设置状态按钮信息：" << bottomName;

    // 使用编译器建议的控件名称
    ui->labelCenter->setStyleSheet(centerPicStyle);
    ui->labelTopName->setStyleSheet(topPicStyle);
    ui->labelTopName->setText(topName);
    ui->labelBottom->setText(bottomName);

    // 保存底部名称
    m_bottomName = bottomName;
}

void MonitoStateW::mousePressEvent(QMouseEvent *event)
{
    // 只处理左键点击
    if (event->button() == Qt::LeftButton) {
        m_isPressed = true;
        qDebug() << "状态按钮被按下：" << m_bottomName;

        // 可以添加按下效果
        setStyleSheet("opacity: 0.8;");
    }

    QWidget::mousePressEvent(event);
}

void MonitoStateW::mouseReleaseEvent(QMouseEvent *event)
{
    // 只处理左键释放
    if (event->button() == Qt::LeftButton && m_isPressed) {
        m_isPressed = false;

        // 恢复原状
        setStyleSheet("");

        qDebug() << "状态按钮被点击：" << m_bottomName;

        // 发送点击信号
        emit sigClicked();
    }

    QWidget::mouseReleaseEvent(event);
}

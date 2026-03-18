#ifndef MONITOSTATEW_H
#define MONITOSTATEW_H

#include <QWidget>
#include <QString>

namespace Ui {
class MonitoStateW;
}

class MonitoStateW : public QWidget
{
    Q_OBJECT

public:
    explicit MonitoStateW(QWidget *parent = nullptr);
    ~MonitoStateW();

    // 设置显示信息
    void setTextInfo(const QString& centerPicStyle, const QString& topPicStyle,
                     const QString& topName, const QString& bottomName);

signals:
    // 添加点击信号
    void sigClicked();

protected:
    // 重写鼠标事件以捕获点击
    void mousePressEvent(QMouseEvent *event) override;
    void mouseReleaseEvent(QMouseEvent *event) override;

private:
    Ui::MonitoStateW *ui;

    // 是否被点击标志
    bool m_isPressed;

    // 保存文本信息
    QString m_bottomName;
};

#endif // MONITOSTATEW_H

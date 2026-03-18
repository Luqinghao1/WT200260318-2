#ifndef MONITORBTN_H
#define MONITORBTN_H

//监控界面按钮
#include <QWidget>

namespace Ui {
class MonitorBtn;
}

class MonitorBtn : public QWidget
{
    Q_OBJECT

public:
    explicit MonitorBtn(QWidget *parent = nullptr);
    ~MonitorBtn();
    void setPicName(QString pic,QString Name);
    void setBtnColorStyle(QString style);
signals:
    void sigClicked(QString name);
protected:
    bool eventFilter(QObject *watched, QEvent *event);

private:
    Ui::MonitorBtn *ui;
    QString m_Style;
};

#endif // MONITORBTN_H

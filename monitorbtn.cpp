#include "monitorbtn.h"
#include "ui_monitorbtn.h"
#include <QMouseEvent>

MonitorBtn::MonitorBtn(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MonitorBtn)
{
    ui->setupUi(this);
    ui->widget->installEventFilter(this);
}

MonitorBtn::~MonitorBtn()
{
    delete ui;
}

void MonitorBtn::setPicName(QString pic, QString Name)
{
    ui->labelPic->setStyleSheet(pic);
    ui->labelName->setText(Name);
}

void MonitorBtn::setBtnColorStyle(QString style)
{
    m_Style = style;
    ui->widget->setStyleSheet(m_Style.arg(120));
}

bool MonitorBtn::eventFilter(QObject *watched, QEvent *event)
{
    if(event->type() == QEvent::Enter)
    {
       ui->widget->setStyleSheet(m_Style.arg(120));

    }
    else if(event->type() == QEvent::Leave)
    {
       ui->widget->setStyleSheet(m_Style.arg(180));
    }
    else if(((QMouseEvent*)event)->button() == Qt::LeftButton)
    {
       ui->widget->setStyleSheet(m_Style.arg(200));
       emit sigClicked(ui->labelName->text());
    }
    return false;
}

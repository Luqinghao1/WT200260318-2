/**
 * @file navbtn.cpp
 * @brief 左侧导航栏按钮控件实现文件
 * @details 实现了导航按钮的初始化、样式设置、事件过滤及信号发射功能。
 * 调整了按钮的高度和尺寸策略以适应更大的图标显示。
 */

#include "navbtn.h"
#include "ui_navbtn.h"
#include <QEvent>

/**
 * @brief 构造函数
 * @param parent 父窗口指针
 */
NavBtn::NavBtn(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::NavBtn)
{
    ui->setupUi(this);

    // 安装事件过滤器，以便在主 Widget 上捕获鼠标点击事件
    ui->widget->installEventFilter(this);

    // 初始化索引
    m_index = 0;

    // --- 界面尺寸设置 ---
    // 设置按钮固定高度为 100 像素，以适应 60x60 的大图标
    // 防止在布局中被拉伸或压缩
    this->setFixedHeight(100);

    // 设置最小宽度，确保文字不被遮挡
    this->setMinimumWidth(110);
}

/**
 * @brief 析构函数
 */
NavBtn::~NavBtn()
{
    delete ui;
}

/**
 * @brief 设置图标和名称
 * @param pic 图标的样式表路径
 * @param Name 按钮显示的文本
 */
void NavBtn::setPicName(QString pic, QString Name)
{
    // 通过样式表设置 QLabel 的背景图片
    ui->labelPic->setStyleSheet(pic);
    // 设置文本标签的内容
    ui->labelName->setText(Name);
}

/**
 * @brief 获取按钮名称
 * @return 按钮文本
 */
QString NavBtn::getName()
{
    return ui->labelName->text();
}

/**
 * @brief 设置按钮索引
 * @param index 索引值
 */
void NavBtn::setIndex(int index)
{
    m_index = index;
}

/**
 * @brief 获取按钮索引
 * @return 索引值
 */
int NavBtn::getIndex()
{
    return m_index;
}

/**
 * @brief 设置为普通样式（未选中）
 */
void NavBtn::setNormalStyle()
{
    // 背景设置为全透明
    ui->widget->setStyleSheet("#widget{background-color: rgb(0,0,0,0);}");
}

/**
 * @brief 设置为选中样式
 */
void NavBtn::setClickedStyle()
{
    // 背景设置为深蓝色半透明，高亮显示当前选中项
    ui->widget->setStyleSheet("#widget{background-color: rgb(27,45,85,100);}");
}

/**
 * @brief 事件过滤器处理
 * @details 拦截内部 widget 的鼠标按下事件，实现点击响应
 * @param watched 被监视的对象
 * @param event 事件对象
 * @return 如果处理了事件返回 true，否则返回 false
 */
bool NavBtn::eventFilter(QObject *watched, QEvent *event)
{
    // 检查是否是鼠标按下事件
    if(event->type() == QEvent::MouseButtonPress)
    {
        // 发送点击信号，携带按钮名称
        emit sigClicked(ui->labelName->text());

        // 立即更新样式为选中状态（视觉反馈）
        ui->widget->setStyleSheet("#widget{background-color: rgb(27,45,85,100);}");
    }
    return false; // 继续传递事件，不阻断后续处理
}

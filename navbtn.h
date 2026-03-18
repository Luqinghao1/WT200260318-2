/**
 * @file navbtn.h
 * @brief 左侧导航栏按钮控件头文件
 * @details 该类定义了主界面左侧导航栏中的自定义按钮。
 * 包含图标显示、文字标签、索引记录以及点击事件的处理。
 * 支持正常状态和选中状态的样式切换。
 */

#ifndef NAVBTN_H
#define NAVBTN_H

#include <QWidget>

// 声明 Ui 命名空间中的 NavBtn 类
namespace Ui {
class NavBtn;
}

/**
 * @class NavBtn
 * @brief 自定义导航按钮控件
 */
class NavBtn : public QWidget
{
    Q_OBJECT

public:
    /**
     * @brief 构造函数
     * @param parent 父窗口指针
     */
    explicit NavBtn(QWidget *parent = nullptr);

    /**
     * @brief 析构函数
     */
    ~NavBtn();

    /**
     * @brief 设置按钮的图标和名称
     * @param pic 图标的样式表字符串（例如：border-image url(...)）
     * @param Name 按钮显示的文字名称
     */
    void setPicName(QString pic, QString Name);

    /**
     * @brief 获取按钮的名称
     * @return 返回按钮显示的文字
     */
    QString getName();

    /**
     * @brief 设置按钮的索引值
     * @param index 索引整数
     */
    void setIndex(int index);

    /**
     * @brief 获取按钮的索引值
     * @return 返回索引整数
     */
    int getIndex();

    /**
     * @brief 设置按钮为普通（未选中）样式
     * @details 背景透明
     */
    void setNormalStyle();

    /**
     * @brief 设置按钮为选中样式
     * @details 背景显示为半透明深蓝色，用于指示当前激活的页面
     */
    void setClickedStyle();

signals:
    /**
     * @brief 点击信号
     * @param name 被点击按钮的名称
     */
    void sigClicked(QString name);

protected:
    /**
     * @brief 事件过滤器
     * @details 用于拦截和处理鼠标点击事件，实现自定义的点击逻辑
     * @param watched 被观察的对象
     * @param event 发生的事件
     * @return bool 是否拦截事件
     */
    bool eventFilter(QObject *watched, QEvent *event);

private:
    Ui::NavBtn *ui; ///< UI 指针，用于访问界面元素
    int m_index;    ///< 按钮的索引，用于标识对应的页面层级
};

#endif // NAVBTN_H

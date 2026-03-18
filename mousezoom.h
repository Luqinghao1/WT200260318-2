#ifndef MOUSEZOOM_H
#define MOUSEZOOM_H

#include "qcustomplot.h"

class MouseZoom : public QCustomPlot
{
    Q_OBJECT

public:
    explicit MouseZoom(QWidget *parent = nullptr);
    ~MouseZoom();

signals:
    // 现有信号保持不变
    void saveImageRequested();
    void exportDataRequested();
    void drawLineRequested(double slope);
    void settingsRequested();
    void resetViewRequested();

    void addAnnotationRequested(QCPItemLine* line);
    void lineStyleRequested(QCPItemLine* line);
    void deleteSelectedRequested();
    void editItemRequested(QCPAbstractItem* item);

protected:
    // [修改] 滚轮事件处理核心缩放逻辑
    void wheelEvent(QWheelEvent *event) override;

    // [新增] 监听键盘按键状态
    void keyPressEvent(QKeyEvent *event) override;
    void keyReleaseEvent(QKeyEvent *event) override;

private slots:
    void onCustomContextMenuRequested(const QPoint &pos);

private:
    double distToSegment(const QPointF& p, const QPointF& s, const QPointF& e);

    // [新增] 记录键盘状态
    bool m_isUpPressed;
    bool m_isDownPressed;
};

#endif // MOUSEZOOM_H

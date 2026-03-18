#include "mousezoom.h"
#include <QApplication>
#include <QMenu>
#include <QAction>
#include <QKeyEvent>
#include <cmath>

MouseZoom::MouseZoom(QWidget *parent)
    : QCustomPlot(parent)
    , m_isUpPressed(false)
    , m_isDownPressed(false)
{
    setInteractions(QCP::iRangeDrag | QCP::iRangeZoom | QCP::iSelectItems);
    setContextMenuPolicy(Qt::CustomContextMenu);
    connect(this, &QCustomPlot::customContextMenuRequested, this, &MouseZoom::onCustomContextMenuRequested);

    // 确保组件能接收键盘事件
    setFocusPolicy(Qt::StrongFocus);
}

MouseZoom::~MouseZoom()
{
}

// 记录键盘按下状态
void MouseZoom::keyPressEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Up) {
        m_isUpPressed = true;
    } else if (event->key() == Qt::Key_Down) {
        m_isDownPressed = true;
    }
    QCustomPlot::keyPressEvent(event);
}

// 记录键盘释放状态
void MouseZoom::keyReleaseEvent(QKeyEvent *event)
{
    if (event->key() == Qt::Key_Up) {
        m_isUpPressed = false;
    } else if (event->key() == Qt::Key_Down) {
        m_isDownPressed = false;
    }
    QCustomPlot::keyReleaseEvent(event);
}

// [核心修改] 滚轮缩放逻辑
void MouseZoom::wheelEvent(QWheelEvent *event)
{
    // 1. 根据按键决定缩放策略
    bool zoomX = true;
    bool zoomY = true;
    bool syncY = true; // 是否强制同步所有坐标系的Y轴

    if (m_isUpPressed) {
        // 模式3: 按住上键 -> 仅纵轴，仅当前坐标系 (不设同步)
        zoomX = false;
        zoomY = true;
        syncY = false;
    } else if (m_isDownPressed) {
        // 模式2: 按住下键 -> 仅横轴 (X轴已通过信号槽自动同步)
        zoomX = true;
        zoomY = false;
        syncY = false;
    } else {
        // 模式1: 直接滚动 -> 横纵同时，且上下Y轴同步
        zoomX = true;
        zoomY = true;
        syncY = true;
    }

    // 2. 配置当前 QCustomPlot 的缩放方向
    // 这决定了 QCustomPlot 自动处理哪个轴的缩放（针对鼠标下的坐标系）

    // [修复] 使用默认构造函数初始化为空，解决编译错误
    Qt::Orientations orient = Qt::Orientations();

    if (zoomX) orient |= Qt::Horizontal;
    if (zoomY) orient |= Qt::Vertical;

    for(int i=0; i<axisRectCount(); ++i) {
        axisRect(i)->setRangeZoom(orient);
    }

    // 3. 处理 Y 轴的“手动同步” (针对模式1)
    // QCustomPlot 默认只缩放鼠标下的坐标系。如果需要上下Y轴同时缩放，
    // 我们需要手动对“非鼠标下”的坐标系应用相同的缩放比例。
    if (zoomY && syncY && axisRectCount() > 1) {
        // 获取鼠标下的坐标系（QCP会自动处理这个）
        QCPAxisRect* currentRect = axisRectAt(event->position().toPoint());

        // 计算缩放因子 (仿照 QCP 内部逻辑)
        // 滚轮向上(delta>0) -> steps > 0 -> factor^steps < 1 -> 范围缩小(放大)
        double wheelSteps = event->angleDelta().y() / 120.0;
        double baseFactor = axisRect(0)->rangeZoomFactor(Qt::Vertical);
        double scale = std::pow(baseFactor, wheelSteps);

        // 对所有【非】当前坐标系的 Y 轴应用缩放
        for(int i=0; i<axisRectCount(); ++i) {
            QCPAxisRect* rect = axisRect(i);
            if (rect != currentRect) {
                // 以当前范围中心为基准进行缩放
                QList<QCPAxis*> axes = rect->axes(QCPAxis::atLeft | QCPAxis::atRight);
                for (QCPAxis* axis : axes) {
                    if (axis->visible()) {
                        axis->scaleRange(scale, axis->range().center());
                    }
                }
            }
        }
    }

    // 4. 调用父类处理标准缩放 (处理鼠标下的坐标系及X轴联动)
    QCustomPlot::wheelEvent(event);

    // (可选) 恢复默认的全向缩放设置，防止影响其他交互
    for(int i=0; i<axisRectCount(); ++i) {
        axisRect(i)->setRangeZoom(Qt::Horizontal | Qt::Vertical);
    }
}

double MouseZoom::distToSegment(const QPointF& p, const QPointF& s, const QPointF& e)
{
    double l2 = (s.x()-e.x())*(s.x()-e.x()) + (s.y()-e.y())*(s.y()-e.y());
    if (l2 == 0) return std::sqrt((p.x()-s.x())*(p.x()-s.x()) + (p.y()-s.y())*(p.y()-s.y()));
    double t = ((p.x()-s.x())*(e.x()-s.x()) + (p.y()-s.y())*(e.y()-s.y())) / l2;
    t = std::max(0.0, std::min(1.0, t));
    QPointF proj = s + t * (e - s);
    return std::sqrt((p.x()-proj.x())*(p.x()-proj.x()) + (p.y()-proj.y())*(p.y()-proj.y()));
}

void MouseZoom::onCustomContextMenuRequested(const QPoint &pos)
{
    QMenu menu(this);

    QCPAbstractItem* hitItem = nullptr;
    QCPItemLine* hitLine = nullptr;
    QCPItemText* hitText = nullptr;
    double tolerance = 8.0;
    QPointF pMouse = pos;

    // Check lines
    for (int i = 0; i < itemCount(); ++i) {
        if (auto line = qobject_cast<QCPItemLine*>(item(i))) {
            if (line->property("isCharacteristic").isValid()) {
                double x1 = xAxis->coordToPixel(line->start->coords().x());
                double y1 = yAxis->coordToPixel(line->start->coords().y());
                double x2 = xAxis->coordToPixel(line->end->coords().x());
                double y2 = yAxis->coordToPixel(line->end->coords().y());
                if (distToSegment(pMouse, QPointF(x1, y1), QPointF(x2, y2)) < tolerance) {
                    hitLine = line;
                    hitItem = line;
                    break;
                }
            }
        }
    }
    // Check text
    if (!hitItem) {
        for (int i = 0; i < itemCount(); ++i) {
            if (auto text = qobject_cast<QCPItemText*>(item(i))) {
                if (text->selectTest(pMouse, false) < tolerance) {
                    hitText = text;
                    hitItem = text;
                    break;
                }
            }
        }
    }

    if (hitLine) {
        deselectAll();
        hitLine->setSelected(true);
        replot();
        QAction* actNote = menu.addAction("添加/修改 标注");
        connect(actNote, &QAction::triggered, this, [=](){ emit addAnnotationRequested(hitLine); });

        QAction* actStyle = menu.addAction("样式设置 (颜色/线型)");
        connect(actStyle, &QAction::triggered, this, [=](){ emit lineStyleRequested(hitLine); });

        menu.addSeparator();
        QAction* actDel = menu.addAction("删除线段");
        connect(actDel, &QAction::triggered, this, &MouseZoom::deleteSelectedRequested);
    }
    else if (hitText) {
        deselectAll();
        hitText->setSelected(true);
        replot();
        QAction* actEdit = menu.addAction("修改标注文字");
        connect(actEdit, &QAction::triggered, this, [=](){ emit editItemRequested(hitText); });
        menu.addSeparator();
        QAction* actDel = menu.addAction("删除标注");
        connect(actDel, &QAction::triggered, this, &MouseZoom::deleteSelectedRequested);
    }
    else {
        QAction* actSave = menu.addAction(QIcon(), "导出图片");
        connect(actSave, &QAction::triggered, this, &MouseZoom::saveImageRequested);
        QAction* actExport = menu.addAction(QIcon(), "导出数据");
        connect(actExport, &QAction::triggered, this, &MouseZoom::exportDataRequested);
        QMenu* subMenuLine = menu.addMenu("标识线绘制");
        QAction* actK1 = subMenuLine->addAction("斜率 k=1");
        QAction* actK05 = subMenuLine->addAction("斜率 k=1/2");
        QAction* actK025 = subMenuLine->addAction("斜率 k=1/4");
        QAction* actK0 = subMenuLine->addAction("水平线");
        connect(actK1, &QAction::triggered, this, [=](){ emit drawLineRequested(1.0); });
        connect(actK05, &QAction::triggered, this, [=](){ emit drawLineRequested(0.5); });
        connect(actK025, &QAction::triggered, this, [=](){ emit drawLineRequested(0.25); });
        connect(actK0, &QAction::triggered, this, [=](){ emit drawLineRequested(0.0); });
        QAction* actSetting = menu.addAction(QIcon(), "图表设置");
        connect(actSetting, &QAction::triggered, this, &MouseZoom::settingsRequested);
        menu.addSeparator();
        QAction* actReset = menu.addAction(QIcon(), "重置视图");
        connect(actReset, &QAction::triggered, this, &MouseZoom::resetViewRequested);
    }

    menu.exec(mapToGlobal(pos));
}

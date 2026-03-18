/*
 * 文件名: plottingdialog1.cpp
 * 文件作用: 新建单一曲线配置对话框实现
 * 功能描述:
 * 1. 实现数据列加载，支持多文件切换联动。
 * 2. 样式设置 UI 优化（纯中文、图标预览）。
 * 3. 默认名称跟随 Y 轴列名变化，遇到特殊字符 '\' 只取前面部分。
 * 4. “显示数据来源”格式为 (文件名)。
 */

#include "plottingdialog1.h"
#include "ui_plottingdialog1.h"
#include <QFileInfo>
#include <QPainter>
#include <QPixmap>

PlottingDialog1::PlottingDialog1(const QMap<QString, QStandardItemModel*>& models, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::PlottingDialog1),
    m_dataMap(models),
    m_currentModel(nullptr)
{
    ui->setupUi(this);

    // 1. 初始化样式控件内容 (填充下拉框，生成图标)
    setupStyleUI();

    // 2. 汉化标准按钮
    ui->buttonBox->button(QDialogButtonBox::Ok)->setText("确定");
    ui->buttonBox->button(QDialogButtonBox::Cancel)->setText("取消");

    // 3. 初始化文件选择下拉框
    ui->comboFileSelect->clear();
    if (m_dataMap.isEmpty()) {
        ui->comboFileSelect->setEnabled(false);
    } else {
        for(auto it = m_dataMap.begin(); it != m_dataMap.end(); ++it) {
            QString filePath = it.key();
            QFileInfo fi(filePath);
            ui->comboFileSelect->addItem(fi.fileName().isEmpty() ? filePath : fi.fileName(), filePath);
        }
    }

    // 4. 信号连接
    connect(ui->comboFileSelect, SIGNAL(currentIndexChanged(int)), this, SLOT(onFileChanged(int)));
    // [新增] 连接 Y 轴变化信号
    connect(ui->combo_YCol, SIGNAL(currentIndexChanged(int)), this, SLOT(onYColumnChanged(int)));
    connect(ui->check_ShowSource, &QCheckBox::toggled, this, &PlottingDialog1::onShowSourceChanged);

    // 5. 触发初始加载
    if (ui->comboFileSelect->count() > 0) {
        ui->comboFileSelect->setCurrentIndex(0);
        onFileChanged(0);
    }
}

PlottingDialog1::~PlottingDialog1()
{
    delete ui;
}

void PlottingDialog1::onFileChanged(int index)
{
    Q_UNUSED(index);
    QString key = ui->comboFileSelect->currentData().toString();

    if (m_dataMap.contains(key)) {
        m_currentModel = m_dataMap.value(key);
    } else {
        m_currentModel = nullptr;
    }
    populateComboBoxes(); // 这里面会触发 YCol 的变化，从而触发 onYColumnChanged 更新名称
}

void PlottingDialog1::populateComboBoxes()
{
    // 暂时阻塞 Y 轴信号，以免清空时触发不必要的逻辑，待填充完毕后再处理
    bool oldState = ui->combo_YCol->blockSignals(true);

    ui->combo_XCol->clear();
    ui->combo_YCol->clear();

    if (m_currentModel) {
        QStringList headers;
        for(int i=0; i<m_currentModel->columnCount(); ++i) {
            QStandardItem* item = m_currentModel->horizontalHeaderItem(i);
            headers << (item ? item->text() : QString("列 %1").arg(i+1));
        }
        ui->combo_XCol->addItems(headers);
        ui->combo_YCol->addItems(headers);

        if(headers.count() > 0) ui->combo_XCol->setCurrentIndex(0);
        if(headers.count() > 1) ui->combo_YCol->setCurrentIndex(1);
    }

    ui->combo_YCol->blockSignals(oldState);

    // 手动调用一次更新逻辑
    onYColumnChanged(ui->combo_YCol->currentIndex());
}

void PlottingDialog1::onYColumnChanged(int index)
{
    Q_UNUSED(index);
    updateBaseName();
    updateNameSuffix();
}

void PlottingDialog1::updateBaseName()
{
    // 获取当前 Y 轴文本
    QString yLabel = ui->combo_YCol->currentText();
    if (yLabel.isEmpty()) return;

    // 规则：只取 '\' 前面的内容
    // split 返回的列表至少包含一个元素（原字符串），所以 first() 是安全的
    QString baseName = yLabel.split('\\').first();

    // 设置到输入框 (暂时不带后缀，后缀由 updateNameSuffix 追加)
    ui->lineEdit_Name->setText(baseName);

    // 清空记录的后缀，因为现在是纯粹的基础名
    m_lastSuffix.clear();
}

void PlottingDialog1::onShowSourceChanged(bool checked)
{
    Q_UNUSED(checked);
    updateNameSuffix();
}

void PlottingDialog1::updateNameSuffix()
{
    // 1. 获取当前文本
    QString currentName = ui->lineEdit_Name->text();

    // 2. 移除上一次添加的后缀（如果存在）
    if (!m_lastSuffix.isEmpty() && currentName.endsWith(m_lastSuffix)) {
        currentName.chop(m_lastSuffix.length());
    }

    // 3. 生成新后缀：(文件名)
    QString newSuffix = "";
    if (ui->check_ShowSource->isChecked()) {
        QString filePath = ui->comboFileSelect->currentData().toString();
        if (!filePath.isEmpty()) {
            QFileInfo fi(filePath);
            newSuffix = QString(" (%1)").arg(fi.completeBaseName());
        }
    }

    // 4. 应用新名称并记录后缀
    ui->lineEdit_Name->setText(currentName + newSuffix);
    m_lastSuffix = newSuffix;
}

void PlottingDialog1::setupStyleUI()
{
    // 1. 填充点形状 (图标 + 纯中文)
    ui->combo_PointShape->clear();
    ui->combo_PointShape->setIconSize(QSize(16, 16));

    ui->combo_PointShape->addItem(createPointIcon(QCPScatterStyle::ssDisc), "实心圆", (int)QCPScatterStyle::ssDisc);
    ui->combo_PointShape->addItem(createPointIcon(QCPScatterStyle::ssCircle), "空心圆", (int)QCPScatterStyle::ssCircle);
    ui->combo_PointShape->addItem(createPointIcon(QCPScatterStyle::ssSquare), "正方形", (int)QCPScatterStyle::ssSquare);
    ui->combo_PointShape->addItem(createPointIcon(QCPScatterStyle::ssDiamond), "菱形", (int)QCPScatterStyle::ssDiamond);
    ui->combo_PointShape->addItem(createPointIcon(QCPScatterStyle::ssTriangle), "三角形", (int)QCPScatterStyle::ssTriangle);
    ui->combo_PointShape->addItem(createPointIcon(QCPScatterStyle::ssCross), "十字", (int)QCPScatterStyle::ssCross);
    ui->combo_PointShape->addItem(createPointIcon(QCPScatterStyle::ssPlus), "加号", (int)QCPScatterStyle::ssPlus);
    ui->combo_PointShape->addItem(createPointIcon(QCPScatterStyle::ssNone), "无", (int)QCPScatterStyle::ssNone);

    // 2. 填充线型 (宽图标 + 纯中文)
    ui->combo_LineStyle->clear();
    ui->combo_LineStyle->setIconSize(QSize(32, 16));

    ui->combo_LineStyle->addItem(createLineIcon(Qt::NoPen), "无", (int)Qt::NoPen);
    ui->combo_LineStyle->addItem(createLineIcon(Qt::SolidLine), "实线", (int)Qt::SolidLine);
    ui->combo_LineStyle->addItem(createLineIcon(Qt::DashLine), "虚线", (int)Qt::DashLine);
    ui->combo_LineStyle->addItem(createLineIcon(Qt::DotLine), "点线", (int)Qt::DotLine);
    ui->combo_LineStyle->addItem(createLineIcon(Qt::DashDotLine), "点划线", (int)Qt::DashDotLine);

    // 3. 填充颜色 (图标 + 纯中文)
    initColorComboBox(ui->combo_PointColor);
    initColorComboBox(ui->combo_LineColor);

    // 4. 设置默认值
    int redIdx = ui->combo_PointColor->findData(QColor(Qt::red));
    if (redIdx != -1) ui->combo_PointColor->setCurrentIndex(redIdx);
    ui->combo_PointShape->setCurrentIndex(0); // 实心圆

    int blueIdx = ui->combo_LineColor->findData(QColor(Qt::blue));
    if (blueIdx != -1) ui->combo_LineColor->setCurrentIndex(blueIdx);
    ui->combo_LineStyle->setCurrentIndex(0); // 无

    ui->spin_LineWidth->setValue(2);
}

void PlottingDialog1::initColorComboBox(QComboBox* combo)
{
    combo->clear();
    combo->setIconSize(QSize(16, 16));

    struct ColorItem { QString name; QColor color; };
    QList<ColorItem> colors = {
        {"黑色", Qt::black}, {"红色", Qt::red}, {"蓝色", Qt::blue},
        {"绿色", Qt::green}, {"青色", Qt::cyan}, {"品红", Qt::magenta},
        {"黄色", Qt::yellow}, {"深红", Qt::darkRed}, {"深绿", Qt::darkGreen},
        {"深蓝", Qt::darkBlue}, {"灰色", Qt::gray}, {"橙色", QColor(255, 165, 0)},
        {"紫色", QColor(128, 0, 128)}, {"棕色", QColor(165, 42, 42)},
        {"粉色", QColor(255, 192, 203)}, {"天蓝", QColor(135, 206, 235)}
    };

    for (const auto& item : colors) {
        QPixmap pix(16, 16);
        pix.fill(item.color);
        QPainter painter(&pix);
        painter.setPen(Qt::gray);
        painter.drawRect(0, 0, 15, 15);
        combo->addItem(QIcon(pix), item.name, item.color);
    }
}

QIcon PlottingDialog1::createPointIcon(QCPScatterStyle::ScatterShape shape)
{
    QPixmap pix(16, 16);
    pix.fill(Qt::transparent);
    QCPPainter painter(&pix);
    painter.setRenderHint(QPainter::Antialiasing);
    QCPScatterStyle ss(shape);
    ss.setPen(QPen(Qt::black));
    ss.setBrush(QBrush(Qt::black));
    ss.setSize(10);
    ss.drawShape(&painter, 8, 8);
    return QIcon(pix);
}

QIcon PlottingDialog1::createLineIcon(Qt::PenStyle style)
{
    QPixmap pix(32, 16);
    pix.fill(Qt::transparent);
    QPainter painter(&pix);
    painter.setRenderHint(QPainter::Antialiasing);

    if (style == Qt::NoPen) {
        painter.setPen(Qt::gray);
        painter.drawText(pix.rect(), Qt::AlignCenter, "无");
    } else {
        QPen pen(Qt::black);
        pen.setStyle(style);
        pen.setWidth(2);
        painter.setPen(pen);
        painter.drawLine(0, 8, 32, 8);
    }
    return QIcon(pix);
}

// --- Getters ---

QString PlottingDialog1::getCurveName() const { return ui->lineEdit_Name->text(); }
QString PlottingDialog1::getLegendName() const { return ui->combo_YCol->currentText(); }
QString PlottingDialog1::getSelectedFileName() const { return ui->comboFileSelect->currentData().toString(); }

int PlottingDialog1::getXColumn() const { return ui->combo_XCol->currentIndex(); }
int PlottingDialog1::getYColumn() const { return ui->combo_YCol->currentIndex(); }
QString PlottingDialog1::getXLabel() const { return ui->combo_XCol->currentText(); }
QString PlottingDialog1::getYLabel() const { return ui->combo_YCol->currentText(); }
bool PlottingDialog1::isNewWindow() const { return ui->check_NewWindow->isChecked(); }

QCPScatterStyle::ScatterShape PlottingDialog1::getPointShape() const {
    return (QCPScatterStyle::ScatterShape)ui->combo_PointShape->currentData().toInt();
}
QColor PlottingDialog1::getPointColor() const {
    return ui->combo_PointColor->currentData().value<QColor>();
}
Qt::PenStyle PlottingDialog1::getLineStyle() const {
    return (Qt::PenStyle)ui->combo_LineStyle->currentData().toInt();
}
QColor PlottingDialog1::getLineColor() const {
    return ui->combo_LineColor->currentData().value<QColor>();
}
int PlottingDialog1::getLineWidth() const {
    return ui->spin_LineWidth->value();
}

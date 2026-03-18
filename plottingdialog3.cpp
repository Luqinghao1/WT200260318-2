/*
 * 文件名: plottingdialog3.cpp
 * 文件作用: 试井分析曲线配置对话框实现
 * 功能描述:
 * 1. 实现了三段式布局逻辑：数据、计算、样式。
 * 2. 实现了初始压力的自动获取逻辑。
 * 3. 样式设置采用了统一的图标+中文风格。
 * 4. 默认名称前缀为“试井分析”。
 * 5. “显示数据来源”格式为 (文件名)。
 */

#include "plottingdialog3.h"
#include "ui_plottingdialog3.h"
#include <QFileInfo>
#include <QPainter>
#include <QPixmap>

int PlottingDialog3::s_counter = 1;

PlottingDialog3::PlottingDialog3(const QMap<QString, QStandardItemModel*>& models, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::PlottingDialog3),
    m_dataMap(models),
    m_currentModel(nullptr)
{
    ui->setupUi(this);

    // 1. 初始化样式控件
    setupStyleUI();

    // 2. 设置默认名称：试井分析 + 数字
    ui->lineEdit_Name->setText(QString("试井分析 %1").arg(s_counter++));

    // 3. 汉化按钮
    ui->buttonBox->button(QDialogButtonBox::Ok)->setText("确定");
    ui->buttonBox->button(QDialogButtonBox::Cancel)->setText("取消");

    // 4. 初始化文件下拉框
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

    // 5. 信号连接
    connect(ui->comboFileSelect, SIGNAL(currentIndexChanged(int)), this, SLOT(onFileChanged(int)));
    connect(ui->comboPress, SIGNAL(currentIndexChanged(int)), this, SLOT(onPressureColumnChanged(int)));

    connect(ui->radioDrawdown, &QRadioButton::toggled, this, &PlottingDialog3::onTestTypeChanged);
    connect(ui->radioBuildup, &QRadioButton::toggled, this, &PlottingDialog3::onTestTypeChanged);

    connect(ui->checkSmooth, &QCheckBox::toggled, this, &PlottingDialog3::onSmoothToggled);
    connect(ui->check_ShowSource, &QCheckBox::toggled, this, &PlottingDialog3::onShowSourceChanged);

    // 6. 初始状态触发
    // 默认压降试井
    ui->radioDrawdown->setChecked(true);
    // 默认不平滑
    ui->checkSmooth->setChecked(false);
    onSmoothToggled(false);

    if (ui->comboFileSelect->count() > 0) {
        ui->comboFileSelect->setCurrentIndex(0);
        onFileChanged(0);
    }
}

PlottingDialog3::~PlottingDialog3()
{
    delete ui;
}

// --- 逻辑处理槽函数 ---

void PlottingDialog3::onFileChanged(int index)
{
    Q_UNUSED(index);
    QString key = ui->comboFileSelect->currentData().toString();
    m_currentModel = m_dataMap.value(key, nullptr);
    populateComboBoxes();

    // 文件改变更新后缀
    updateNameSuffix();
}

void PlottingDialog3::onShowSourceChanged(bool checked)
{
    Q_UNUSED(checked);
    updateNameSuffix();
}

void PlottingDialog3::updateNameSuffix()
{
    // 1. 获取当前文本
    QString currentName = ui->lineEdit_Name->text();

    // 2. 移除上一次添加的后缀
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

    // 4. 应用
    ui->lineEdit_Name->setText(currentName + newSuffix);
    m_lastSuffix = newSuffix;
}

void PlottingDialog3::populateComboBoxes()
{
    ui->comboTime->clear();
    ui->comboPress->clear();

    if (!m_currentModel) return;

    QStringList headers;
    for(int i=0; i<m_currentModel->columnCount(); ++i) {
        QStandardItem* item = m_currentModel->horizontalHeaderItem(i);
        headers << (item ? item->text() : QString("列 %1").arg(i+1));
    }
    ui->comboTime->addItems(headers);
    ui->comboPress->addItems(headers);

    if(headers.count() > 0) ui->comboTime->setCurrentIndex(0);
    if(headers.count() > 1) ui->comboPress->setCurrentIndex(1);
}

void PlottingDialog3::onPressureColumnChanged(int index)
{
    Q_UNUSED(index);
    updateInitialPressureDefault();
}

void PlottingDialog3::onTestTypeChanged()
{
    bool isDrawdown = ui->radioDrawdown->isChecked();
    ui->spinPi->setEnabled(isDrawdown);
    ui->labelPi->setEnabled(isDrawdown);

    if (isDrawdown) {
        updateInitialPressureDefault();
    }
}

void PlottingDialog3::updateInitialPressureDefault()
{
    if (!ui->radioDrawdown->isChecked()) return;
    if (!m_currentModel) return;

    int col = ui->comboPress->currentIndex();
    if (col >= 0 && m_currentModel->rowCount() > 0) {
        double val = m_currentModel->item(0, col)->text().toDouble();
        ui->spinPi->setValue(val);
    }
}

void PlottingDialog3::onSmoothToggled(bool checked)
{
    ui->labelSmoothFactor->setEnabled(checked);
    ui->spinSmooth->setEnabled(checked);
}

// --- 样式 UI 初始化 ---

void PlottingDialog3::setupStyleUI()
{
    // 1. 准备下拉框列表
    QList<QComboBox*> shapeCombos = {ui->comboPressShape, ui->comboDerivShape};
    QList<QComboBox*> lineCombos = {ui->comboPressLine, ui->comboDerivLine};
    QList<QComboBox*> colorCombos = {
        ui->comboPressPointColor, ui->comboPressLineColor,
        ui->comboDerivPointColor, ui->comboDerivLineColor
    };

    // 2. 填充点形状
    for(auto cb : shapeCombos) {
        cb->clear();
        cb->setIconSize(QSize(16, 16));
        cb->addItem(createPointIcon(QCPScatterStyle::ssDisc), "实心圆", (int)QCPScatterStyle::ssDisc);
        cb->addItem(createPointIcon(QCPScatterStyle::ssCircle), "空心圆", (int)QCPScatterStyle::ssCircle);
        cb->addItem(createPointIcon(QCPScatterStyle::ssSquare), "正方形", (int)QCPScatterStyle::ssSquare);
        cb->addItem(createPointIcon(QCPScatterStyle::ssDiamond), "菱形", (int)QCPScatterStyle::ssDiamond);
        cb->addItem(createPointIcon(QCPScatterStyle::ssTriangle), "三角形", (int)QCPScatterStyle::ssTriangle);
        cb->addItem(createPointIcon(QCPScatterStyle::ssCross), "十字", (int)QCPScatterStyle::ssCross);
        cb->addItem(createPointIcon(QCPScatterStyle::ssPlus), "加号", (int)QCPScatterStyle::ssPlus);
        cb->addItem(createPointIcon(QCPScatterStyle::ssNone), "无", (int)QCPScatterStyle::ssNone);
    }

    // 3. 填充线型
    for(auto cb : lineCombos) {
        cb->clear();
        cb->setIconSize(QSize(32, 16));
        cb->addItem(createLineIcon(Qt::NoPen), "无", (int)Qt::NoPen);
        cb->addItem(createLineIcon(Qt::SolidLine), "实线", (int)Qt::SolidLine);
        cb->addItem(createLineIcon(Qt::DashLine), "虚线", (int)Qt::DashLine);
        cb->addItem(createLineIcon(Qt::DotLine), "点线", (int)Qt::DotLine);
        cb->addItem(createLineIcon(Qt::DashDotLine), "点划线", (int)Qt::DashDotLine);
    }

    // 4. 填充颜色
    for(auto cb : colorCombos) {
        initColorComboBox(cb);
    }

    // 5. 设置默认值
    int redIdx = ui->comboPressPointColor->findData(QColor(Qt::red));
    if(redIdx != -1) {
        ui->comboPressPointColor->setCurrentIndex(redIdx);
        ui->comboPressLineColor->setCurrentIndex(redIdx);
    }
    ui->comboPressShape->setCurrentIndex(0);
    ui->comboPressLine->setCurrentIndex(0);
    ui->spinPressLineWidth->setValue(2);

    int blueIdx = ui->comboDerivPointColor->findData(QColor(Qt::blue));
    if(blueIdx != -1) {
        ui->comboDerivPointColor->setCurrentIndex(blueIdx);
        ui->comboDerivLineColor->setCurrentIndex(blueIdx);
    }
    ui->comboDerivShape->setCurrentIndex(4);
    ui->comboDerivLine->setCurrentIndex(0);
    ui->spinDerivLineWidth->setValue(2);
}

void PlottingDialog3::initColorComboBox(QComboBox* combo) {
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

QIcon PlottingDialog3::createPointIcon(QCPScatterStyle::ScatterShape shape) {
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

QIcon PlottingDialog3::createLineIcon(Qt::PenStyle style) {
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

QString PlottingDialog3::getCurveName() const { return ui->lineEdit_Name->text(); }
QString PlottingDialog3::getSelectedFileName() const { return ui->comboFileSelect->currentData().toString(); }

int PlottingDialog3::getTimeColumn() const { return ui->comboTime->currentIndex(); }
int PlottingDialog3::getPressureColumn() const { return ui->comboPress->currentIndex(); }

PlottingDialog3::TestType PlottingDialog3::getTestType() const {
    return ui->radioDrawdown->isChecked() ? Drawdown : Buildup;
}
double PlottingDialog3::getInitialPressure() const { return ui->spinPi->value(); }
double PlottingDialog3::getLSpacing() const { return ui->spinL->value(); }
bool PlottingDialog3::isSmoothEnabled() const { return ui->checkSmooth->isChecked(); }
int PlottingDialog3::getSmoothFactor() const { return ui->spinSmooth->value(); }

QCPScatterStyle::ScatterShape PlottingDialog3::getPressShape() const {
    return (QCPScatterStyle::ScatterShape)ui->comboPressShape->currentData().toInt();
}
QColor PlottingDialog3::getPressPointColor() const {
    return ui->comboPressPointColor->currentData().value<QColor>();
}
Qt::PenStyle PlottingDialog3::getPressLineStyle() const {
    return (Qt::PenStyle)ui->comboPressLine->currentData().toInt();
}
QColor PlottingDialog3::getPressLineColor() const {
    return ui->comboPressLineColor->currentData().value<QColor>();
}
int PlottingDialog3::getPressLineWidth() const { return ui->spinPressLineWidth->value(); }

QCPScatterStyle::ScatterShape PlottingDialog3::getDerivShape() const {
    return (QCPScatterStyle::ScatterShape)ui->comboDerivShape->currentData().toInt();
}
QColor PlottingDialog3::getDerivPointColor() const {
    return ui->comboDerivPointColor->currentData().value<QColor>();
}
Qt::PenStyle PlottingDialog3::getDerivLineStyle() const {
    return (Qt::PenStyle)ui->comboDerivLine->currentData().toInt();
}
QColor PlottingDialog3::getDerivLineColor() const {
    return ui->comboDerivLineColor->currentData().value<QColor>();
}
int PlottingDialog3::getDerivLineWidth() const { return ui->spinDerivLineWidth->value(); }

bool PlottingDialog3::isNewWindow() const { return ui->check_NewWindow->isChecked(); }

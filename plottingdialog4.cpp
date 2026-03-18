/*
 * 文件名: plottingdialog4.cpp
 * 文件作用: 曲线管理与修改对话框实现
 * 功能描述:
 * 1. 修复了双栏布局中左侧控件无数据的问题（手动填充 _Dup 控件）。
 * 2. 优化了数据同步逻辑，确保文件切换时列选项正确更新。
 * 3. 实现了样式图标化和左右栏等宽布局逻辑（通过 C++ 代码设置 stretch）。
 */

#include "plottingdialog4.h"
#include "ui_plottingdialog4.h"
#include <QFileInfo>
#include <QPainter>
#include <QDebug>

PlottingDialog4::PlottingDialog4(const QMap<QString, QStandardItemModel*>& models, QWidget *parent) :
    QDialog(parent),
    ui(new Ui::PlottingDialog4),
    m_dataMap(models),
    m_currentType(0)
{
    ui->setupUi(this);

    // [核心修复] 通过代码设置布局比例，解决 UI 文件编译错误，并实现左右等宽
    // Index 0: Left Layout (Stretch 1)
    // Index 1: Middle Line (Stretch 0)
    // Index 2: Right Layout (Stretch 1)
    ui->hboxDual->setStretch(0, 1);
    ui->hboxDual->setStretch(1, 0);
    ui->hboxDual->setStretch(2, 1);

    ui->hboxDualStyle->setStretch(0, 1);
    ui->hboxDualStyle->setStretch(1, 0);
    ui->hboxDualStyle->setStretch(2, 1);

    // 1. 初始化文件下拉框 (同时填充 Main 和 Main_Dup)
    ui->comboFile_1->clear();
    ui->comboFile_1_Dup->clear();
    ui->comboFile_2->clear();

    for(auto it = m_dataMap.begin(); it != m_dataMap.end(); ++it) {
        QString filePath = it.key();
        QFileInfo fi(filePath);
        QString name = fi.fileName().isEmpty() ? filePath : fi.fileName();

        ui->comboFile_1->addItem(name, filePath);
        ui->comboFile_1_Dup->addItem(name, filePath); // 填充副本
        ui->comboFile_2->addItem(name, filePath);
    }

    // 2. 初始化样式下拉框内容
    setupStyleUI();

    // 3. 信号连接
    connect(ui->comboFile_1, SIGNAL(currentIndexChanged(int)), this, SLOT(onFile1Changed(int)));
    connect(ui->comboFile_1_Dup, SIGNAL(currentIndexChanged(int)), this, SLOT(onFile1DupChanged(int)));
    connect(ui->comboFile_2, SIGNAL(currentIndexChanged(int)), this, SLOT(onFile2Changed(int)));

    // Type 1: 产量类型切换
    connect(ui->comboProdType, SIGNAL(currentIndexChanged(int)), this, SLOT(onProdTypeChanged(int)));

    // Type 2: 计算设置
    connect(ui->radioDrawdown, &QRadioButton::toggled, this, &PlottingDialog4::onTestTypeChanged);
    connect(ui->radioBuildup, &QRadioButton::toggled, this, &PlottingDialog4::onTestTypeChanged);
    connect(ui->checkSmooth, &QCheckBox::toggled, this, &PlottingDialog4::onSmoothToggled);

    // 按钮汉化
    ui->buttonBox->button(QDialogButtonBox::Ok)->setText("确定");
    ui->buttonBox->button(QDialogButtonBox::Cancel)->setText("取消");
}

PlottingDialog4::~PlottingDialog4()
{
    delete ui;
}

void PlottingDialog4::initialize(const DialogCurveInfo& info)
{
    m_currentType = info.type;
    ui->lineEdit_Name->setText(info.name);

    // --- 1. 界面布局切换与标签设置 ---
    if (m_currentType == 0) { // 通用曲线
        ui->stackData->setCurrentIndex(0); // 单数据
        ui->groupCalc->setVisible(false);
        ui->stackStyle->setCurrentIndex(0); // 单样式

        ui->label_X1->setText("X轴数据:");
        ui->label_Y1->setText("Y轴数据:");
    }
    else if (m_currentType == 1) { // 压力产量
        ui->stackData->setCurrentIndex(1); // 双数据
        ui->groupCalc->setVisible(false);
        ui->stackStyle->setCurrentIndex(1); // 双样式
        ui->stackStyleRight->setCurrentIndex(0); // 产量样式页

        ui->titleP->setText("压力数据");
        ui->titleQ->setText("产量数据");
        ui->titleS1->setText("压力曲线样式");
        ui->titleS2->setText("产量曲线样式");
    }
    else if (m_currentType == 2) { // 压力导数
        ui->stackData->setCurrentIndex(0); // 单数据 (用于时间/压力)
        ui->groupCalc->setVisible(true);
        ui->stackStyle->setCurrentIndex(1); // 双样式
        ui->stackStyleRight->setCurrentIndex(1); // 导数样式页

        ui->label_X1->setText("时间数据:");
        ui->label_Y1->setText("压力数据:");

        ui->titleS1->setText("压差曲线样式");
        ui->titleS3->setText("导数曲线样式");
    }

    // --- 2. 数据回显 ---

    // Main Data (Type 0/2 use comboFile_1, Type 1 uses comboFile_1_Dup)
    if (m_currentType == 1) {
        int fileIdx = ui->comboFile_1_Dup->findData(info.sourceFileName);
        if(fileIdx != -1) ui->comboFile_1_Dup->setCurrentIndex(fileIdx);
        // 手动触发一次填充，确保列存在
        populateColumns(info.sourceFileName, ui->comboX_1, ui->comboY_1, ui->comboX_1_Dup, ui->comboY_1_Dup);
        if(info.xCol >= 0) ui->comboX_1_Dup->setCurrentIndex(info.xCol);
        if(info.yCol >= 0) ui->comboY_1_Dup->setCurrentIndex(info.yCol);
    } else {
        int fileIdx = ui->comboFile_1->findData(info.sourceFileName);
        if(fileIdx != -1) ui->comboFile_1->setCurrentIndex(fileIdx);
        populateColumns(info.sourceFileName, ui->comboX_1, ui->comboY_1, ui->comboX_1_Dup, ui->comboY_1_Dup);
        if(info.xCol >= 0) ui->comboX_1->setCurrentIndex(info.xCol);
        if(info.yCol >= 0) ui->comboY_1->setCurrentIndex(info.yCol);
    }

    // Main Style (Type 0 uses _1, Type 1/2 uses _1_Dup)
    // 为简化，两个控件都设置
    auto setCombo = [](QComboBox* cb, int val) {
        int idx = cb->findData(val);
        if(idx != -1) cb->setCurrentIndex(idx);
    };
    auto setComboColor = [](QComboBox* cb, QColor val) {
        int idx = cb->findData(val);
        if(idx != -1) cb->setCurrentIndex(idx);
    };

    // 单一样式页
    setCombo(ui->comboShape_1, (int)info.pointShape);
    setComboColor(ui->comboPointColor_1, info.pointColor);
    setCombo(ui->comboLineStyle_1, (int)info.lineStyle);
    setComboColor(ui->comboLineColor_1, info.lineColor);
    ui->spinWidth_1->setValue(info.lineWidth);

    // 双样式页左侧 (副本)
    setCombo(ui->comboShape_1_Dup, (int)info.pointShape);
    setComboColor(ui->comboPointColor_1_Dup, info.pointColor);
    setCombo(ui->comboLineStyle_1_Dup, (int)info.lineStyle);
    setComboColor(ui->comboLineColor_1_Dup, info.lineColor);
    ui->spinWidth_1_Dup->setValue(info.lineWidth);

    // Second Data & Style (Type 1 & 2)
    if (m_currentType == 1) {
        // Data 2
        int fileIdx2 = ui->comboFile_2->findData(info.sourceFileName2);
        if(fileIdx2 != -1) ui->comboFile_2->setCurrentIndex(fileIdx2);
        populateColumns(info.sourceFileName2, ui->comboX_2, ui->comboY_2); // Only for Right
        if(info.x2Col >= 0) ui->comboX_2->setCurrentIndex(info.x2Col);
        if(info.y2Col >= 0) ui->comboY_2->setCurrentIndex(info.y2Col);

        // Prod Style
        setCombo(ui->comboProdType, info.prodGraphType);
        onProdTypeChanged(ui->comboProdType->currentIndex());

        setCombo(ui->comboProdShape, (int)info.style2PointShape);
        setComboColor(ui->comboProdPointColor, info.style2PointColor);
        setCombo(ui->comboProdLineStyle, (int)info.style2LineStyle);
        setComboColor(ui->comboProdLineColor, info.style2LineColor);
        ui->spinProdWidth->setValue(info.style2LineWidth);
    }
    else if (m_currentType == 2) {
        // Calculation
        if(info.testType == 0) ui->radioDrawdown->setChecked(true);
        else ui->radioBuildup->setChecked(true);
        ui->spinPi->setValue(info.initialPressure);
        ui->spinL->setValue(info.LSpacing);
        ui->checkSmooth->setChecked(info.isSmooth);
        ui->spinSmooth->setValue(info.smoothFactor);
        onTestTypeChanged();
        onSmoothToggled(info.isSmooth);

        // Deriv Style
        setCombo(ui->comboDerivShape, (int)info.style2PointShape);
        setComboColor(ui->comboDerivPointColor, info.style2PointColor);
        setCombo(ui->comboDerivLineStyle, (int)info.style2LineStyle);
        setComboColor(ui->comboDerivLineColor, info.style2LineColor);
        ui->spinDerivWidth->setValue(info.style2LineWidth);
    }
}

DialogCurveInfo PlottingDialog4::getResult() const
{
    DialogCurveInfo info;
    info.type = m_currentType;
    info.name = ui->lineEdit_Name->text();

    // Data 1 & Style 1 Retrieval
    if (m_currentType == 1) { // 使用副本控件
        info.sourceFileName = ui->comboFile_1_Dup->currentData().toString();
        info.xCol = ui->comboX_1_Dup->currentIndex();
        info.yCol = ui->comboY_1_Dup->currentIndex();

        info.pointShape = (QCPScatterStyle::ScatterShape)ui->comboShape_1_Dup->currentData().toInt();
        info.pointColor = ui->comboPointColor_1_Dup->currentData().value<QColor>();
        info.lineStyle = (Qt::PenStyle)ui->comboLineStyle_1_Dup->currentData().toInt();
        info.lineColor = ui->comboLineColor_1_Dup->currentData().value<QColor>();
        info.lineWidth = ui->spinWidth_1_Dup->value();
    }
    else if (m_currentType == 2) { // 使用原始数据控件，但使用副本样式控件(左侧)
        info.sourceFileName = ui->comboFile_1->currentData().toString();
        info.xCol = ui->comboX_1->currentIndex();
        info.yCol = ui->comboY_1->currentIndex();

        // 样式取自左侧副本栏 (压差)
        info.pointShape = (QCPScatterStyle::ScatterShape)ui->comboShape_1_Dup->currentData().toInt();
        info.pointColor = ui->comboPointColor_1_Dup->currentData().value<QColor>();
        info.lineStyle = (Qt::PenStyle)ui->comboLineStyle_1_Dup->currentData().toInt();
        info.lineColor = ui->comboLineColor_1_Dup->currentData().value<QColor>();
        info.lineWidth = ui->spinWidth_1_Dup->value();
    }
    else { // 通用 (Type 0)
        info.sourceFileName = ui->comboFile_1->currentData().toString();
        info.xCol = ui->comboX_1->currentIndex();
        info.yCol = ui->comboY_1->currentIndex();

        info.pointShape = (QCPScatterStyle::ScatterShape)ui->comboShape_1->currentData().toInt();
        info.pointColor = ui->comboPointColor_1->currentData().value<QColor>();
        info.lineStyle = (Qt::PenStyle)ui->comboLineStyle_1->currentData().toInt();
        info.lineColor = ui->comboLineColor_1->currentData().value<QColor>();
        info.lineWidth = ui->spinWidth_1->value();
    }

    if (m_currentType == 1) {
        // Data 2
        info.sourceFileName2 = ui->comboFile_2->currentData().toString();
        info.x2Col = ui->comboX_2->currentIndex();
        info.y2Col = ui->comboY_2->currentIndex();

        // Prod Style
        info.prodGraphType = ui->comboProdType->currentData().toInt();
        info.style2PointShape = (QCPScatterStyle::ScatterShape)ui->comboProdShape->currentData().toInt();
        info.style2PointColor = ui->comboProdPointColor->currentData().value<QColor>();
        info.style2LineStyle = (Qt::PenStyle)ui->comboProdLineStyle->currentData().toInt();
        info.style2LineColor = ui->comboProdLineColor->currentData().value<QColor>();
        info.style2LineWidth = ui->spinProdWidth->value();
    }
    else if (m_currentType == 2) {
        // Calculation
        info.testType = ui->radioDrawdown->isChecked() ? 0 : 1;
        info.initialPressure = ui->spinPi->value();
        info.LSpacing = ui->spinL->value();
        info.isSmooth = ui->checkSmooth->isChecked();
        info.smoothFactor = ui->spinSmooth->value();

        // Deriv Style
        info.style2PointShape = (QCPScatterStyle::ScatterShape)ui->comboDerivShape->currentData().toInt();
        info.style2PointColor = ui->comboDerivPointColor->currentData().value<QColor>();
        info.style2LineStyle = (Qt::PenStyle)ui->comboDerivLineStyle->currentData().toInt();
        info.style2LineColor = ui->comboDerivLineColor->currentData().value<QColor>();
        info.style2LineWidth = ui->spinDerivWidth->value();
    }

    return info;
}

// --- 逻辑槽函数 ---

void PlottingDialog4::onFile1Changed(int index) {
    Q_UNUSED(index);
    populateColumns(ui->comboFile_1->currentData().toString(), ui->comboX_1, ui->comboY_1, ui->comboX_1_Dup, ui->comboY_1_Dup);
}

void PlottingDialog4::onFile1DupChanged(int index) {
    Q_UNUSED(index);
    // 同样刷新所有关联列，保持一致
    populateColumns(ui->comboFile_1_Dup->currentData().toString(), ui->comboX_1, ui->comboY_1, ui->comboX_1_Dup, ui->comboY_1_Dup);
}

void PlottingDialog4::onFile2Changed(int index) {
    Q_UNUSED(index);
    populateColumns(ui->comboFile_2->currentData().toString(), ui->comboX_2, ui->comboY_2);
}

void PlottingDialog4::populateColumns(QString key, QComboBox* xCombo, QComboBox* yCombo, QComboBox* xComboDup, QComboBox* yComboDup) {
    if (xCombo) xCombo->clear();
    if (yCombo) yCombo->clear();
    if (xComboDup) xComboDup->clear();
    if (yComboDup) yComboDup->clear();

    if(m_dataMap.contains(key)) {
        QStandardItemModel* model = m_dataMap.value(key);
        QStringList headers;
        for(int i=0; i<model->columnCount(); ++i) {
            QStandardItem* item = model->horizontalHeaderItem(i);
            headers << (item ? item->text() : QString("列 %1").arg(i+1));
        }
        if (xCombo) xCombo->addItems(headers);
        if (yCombo) yCombo->addItems(headers);
        if (xComboDup) xComboDup->addItems(headers);
        if (yComboDup) yComboDup->addItems(headers);
    }
}

void PlottingDialog4::onProdTypeChanged(int index) {
    int type = ui->comboProdType->itemData(index).toInt();
    bool showPoint = (type == 2); // Scatter
    bool showLine = true;

    ui->label_ProdShape->setVisible(showPoint);
    ui->comboProdShape->setVisible(showPoint);
    ui->label_ProdPointColor->setVisible(showPoint);
    ui->comboProdPointColor->setVisible(showPoint);

    ui->label_ProdLineStyle->setVisible(showLine);
    ui->comboProdLineStyle->setVisible(showLine);
    ui->label_ProdLineColor->setVisible(showLine);
    ui->comboProdLineColor->setVisible(showLine);
    ui->label_ProdWidth->setVisible(showLine);
    ui->spinProdWidth->setVisible(showLine);
}

void PlottingDialog4::onTestTypeChanged() {
    bool isDrawdown = ui->radioDrawdown->isChecked();
    ui->label_Pi->setEnabled(isDrawdown);
    ui->spinPi->setEnabled(isDrawdown);
}

void PlottingDialog4::onSmoothToggled(bool checked) {
    ui->label_SmoothFactor->setEnabled(checked);
    ui->spinSmooth->setEnabled(checked);
}

// --- 样式初始化 ---

void PlottingDialog4::setupStyleUI() {
    // 必须包含所有 _Dup 控件，否则它们是空的无法选择
    QList<QComboBox*> shapeCombos = {
        ui->comboShape_1, ui->comboShape_1_Dup, // 添加了副本
        ui->comboProdShape, ui->comboDerivShape
    };
    QList<QComboBox*> lineCombos = {
        ui->comboLineStyle_1, ui->comboLineStyle_1_Dup, // 添加了副本
        ui->comboProdLineStyle, ui->comboDerivLineStyle
    };
    QList<QComboBox*> colorCombos = {
        ui->comboPointColor_1, ui->comboPointColor_1_Dup, // 添加了副本
        ui->comboLineColor_1, ui->comboLineColor_1_Dup,   // 添加了副本
        ui->comboProdPointColor, ui->comboProdLineColor,
        ui->comboDerivPointColor, ui->comboDerivLineColor
    };

    // 填充内容
    for(auto cb : shapeCombos) {
        cb->clear(); cb->setIconSize(QSize(16,16));
        cb->addItem(createPointIcon(QCPScatterStyle::ssDisc), "实心圆", (int)QCPScatterStyle::ssDisc);
        cb->addItem(createPointIcon(QCPScatterStyle::ssCircle), "空心圆", (int)QCPScatterStyle::ssCircle);
        cb->addItem(createPointIcon(QCPScatterStyle::ssSquare), "正方形", (int)QCPScatterStyle::ssSquare);
        cb->addItem(createPointIcon(QCPScatterStyle::ssDiamond), "菱形", (int)QCPScatterStyle::ssDiamond);
        cb->addItem(createPointIcon(QCPScatterStyle::ssTriangle), "三角形", (int)QCPScatterStyle::ssTriangle);
        cb->addItem(createPointIcon(QCPScatterStyle::ssCross), "十字", (int)QCPScatterStyle::ssCross);
        cb->addItem(createPointIcon(QCPScatterStyle::ssPlus), "加号", (int)QCPScatterStyle::ssPlus);
        cb->addItem(createPointIcon(QCPScatterStyle::ssNone), "无", (int)QCPScatterStyle::ssNone);
    }
    for(auto cb : lineCombos) {
        cb->clear(); cb->setIconSize(QSize(32,16));
        cb->addItem(createLineIcon(Qt::NoPen), "无", (int)Qt::NoPen);
        cb->addItem(createLineIcon(Qt::SolidLine), "实线", (int)Qt::SolidLine);
        cb->addItem(createLineIcon(Qt::DashLine), "虚线", (int)Qt::DashLine);
        cb->addItem(createLineIcon(Qt::DotLine), "点线", (int)Qt::DotLine);
        cb->addItem(createLineIcon(Qt::DashDotLine), "点划线", (int)Qt::DashDotLine);
    }
    for(auto cb : colorCombos) initColorComboBox(cb);

    // 填充产量类型
    ui->comboProdType->clear();
    ui->comboProdType->addItem("阶梯图", 0);
    ui->comboProdType->addItem("折线图", 1);
    ui->comboProdType->addItem("散点图", 2);
}

void PlottingDialog4::initColorComboBox(QComboBox* combo) {
    combo->clear(); combo->setIconSize(QSize(16,16));
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
        QPixmap pix(16, 16); pix.fill(item.color);
        QPainter painter(&pix); painter.setPen(Qt::gray); painter.drawRect(0, 0, 15, 15);
        combo->addItem(QIcon(pix), item.name, item.color);
    }
}

QIcon PlottingDialog4::createPointIcon(QCPScatterStyle::ScatterShape shape) {
    QPixmap pix(16, 16); pix.fill(Qt::transparent);
    QCPPainter painter(&pix); painter.setRenderHint(QPainter::Antialiasing);
    QCPScatterStyle ss(shape); ss.setPen(QPen(Qt::black)); ss.setBrush(QBrush(Qt::black));
    ss.setSize(10); ss.drawShape(&painter, 8, 8);
    return QIcon(pix);
}

QIcon PlottingDialog4::createLineIcon(Qt::PenStyle style) {
    QPixmap pix(32, 16); pix.fill(Qt::transparent);
    QPainter painter(&pix); painter.setRenderHint(QPainter::Antialiasing);
    if (style == Qt::NoPen) { painter.setPen(Qt::gray); painter.drawText(pix.rect(), Qt::AlignCenter, "无"); }
    else { QPen pen(Qt::black); pen.setStyle(style); pen.setWidth(2); painter.setPen(pen); painter.drawLine(0, 8, 32, 8); }
    return QIcon(pix);
}

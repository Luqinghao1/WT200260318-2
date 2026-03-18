#ifndef NEWPROJECTDIALOG_H
#define NEWPROJECTDIALOG_H

#include <QDialog>
#include <QJsonObject>
#include <QDateTime>

namespace Ui {
class NewProjectDialog;
}

// 项目单位制枚举
enum class ProjectUnitType {
    Metric_SI = 0, // 公制
    Field_Unit = 1 // 英制/矿场单位
};

// 项目数据结构体
struct ProjectData {
    // --- 基本信息 ---
    QString projectName;    // 项目名
    QString oilFieldName;   // 油田名
    QString wellName;       // 井名
    QString engineer;       // 工程师
    QString comments;       // 备注
    QString projectPath;    // 项目保存的根目录
    QString fullFilePath;   // 生成的具体项目文件全路径 (.wtproj)

    // --- 第二页：井参数 / 油藏参数 ---
    int testType;           // 0: 压力降落, 1: 压力恢复

    // [新增] 水平井长度和裂缝条数
    double horizLength;     // L
    double fracCount;       // nf

    double productionRate;  // 测试产量
    double porosity;        // 孔隙度
    double thickness;       // 有效厚度
    double wellRadius;      // 井筒半径

    // --- 第三页：PVT 参数 ---
    QDateTime testDate;     // 测试日期
    double compressibility; // 综合压缩系数
    double viscosity;       // 粘度
    double volumeFactor;    // 体积系数

    // 界面选择的单位制
    ProjectUnitType currentUnitSystem;
};

class NewProjectDialog : public QDialog
{
    Q_OBJECT

public:
    explicit NewProjectDialog(QWidget *parent = nullptr);
    ~NewProjectDialog();

    // 获取最终生成的项目数据
    ProjectData getProjectData() const;

private slots:
    void on_btnBrowse_clicked();
    void on_comboUnits_currentIndexChanged(int index);
    void on_btnOk_clicked();
    void on_btnCancel_clicked();

private:
    Ui::NewProjectDialog *ui;
    ProjectData m_projectData;

    void initDefaultValues();
    void updateUnitLabels(ProjectUnitType unit);
    void convertValues(ProjectUnitType from, ProjectUnitType to);
    bool createProjectStructure();

    // 加载美化样式表 (修复日期控件文字颜色)
    void loadModernStyle();

    void saveProjectFile(const QString &filePath, const ProjectData &data);
};

#endif // NEWPROJECTDIALOG_H

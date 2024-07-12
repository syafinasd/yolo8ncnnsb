#include "yoloV8.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <algorithm>

// Implementasi fungsi-fungsi yang diperlukan seperti yang sudah dijelaskan di atas

const char* class_names[] = {
    "Adem Sari Panas Dalam", "Ahh Rasa Roti Goguma Korea", "Azarine Acne Gentle Cleansing Foam",
    "Azarine Sunscreen Gel", "Barakat Pasta Gigi", "Biore Pore Pack with Charcoal", "Biore UV", 
    "Cleo Air Mineral", "Club Air Mineral", "Eiem Beauty Natural Sunscreen", 
    "Emeron Lovely White UV Jeju Orange", "Emina Bright Stuff Face Wash", "Emina Sun Protection", 
    "Energen Rasa Coklat", "Eternalplus", "Focallure 3 in 1 Palette", "FreshCare Hot", 
    "Gimbori Rumput Laut Kering Tabur", "Hot In Cream", "Implora Jelly Tint", 
    "Indomie Rasa Kaldu Ayam", "Indomilk Korean Black Latte", "Le Minerale", 
    "Makarizo Hair Energy Morning Dew", "Master Potato Crisps Rendang", 
    "Milku Susu UHT Rasa Coklat Premium", "Milku Susu UHT Rasa Stroberi", "Mixagrip Flu - Batuk", 
    "Nano Nano Milky Stroberi", "Neo Coffee", "Pop Ice Rasa Anggur", "Pop Ice Rasa Cokelat", 
    "Pop Ice Rasa Taro", "Prochiz Spready Keju Oles", "Promag Herbal", "Qlife Menstrual Care", 
    "Richeese Pasta Keju", "Skintific Sunscreen Mist", "Soffell Wangi Apel", "Standart Pen AE7", 
    "Teh Javana", "Tolakangin", "Tolakangin Anak", "Ultramilk Rasa Taro", "Vape Mat 4.1 MV", 
    "Vit Air Mineral", "Vital Ear Oil", "Vitamin B", "Yplogy Hair Mask Gingseng", "Zen Antibacterial"
};

// Helper functions and methods as previously described
// ...

YoloV8::YoloV8() { /* constructor implementation */ }

int YoloV8::load(int _target_size)
{
    yolo.clear();
    yolo.opt = ncnn::Option();
    yolo.opt.num_threads = 4;
    yolo.load_param("./model.ncnn.param");
    yolo.load_model("./model.ncnn.bin");

    target_size = _target_size;
    mean_vals[0] = 103.53f;
    mean_vals[1] = 116.28f;
    mean_vals[2] = 123.675f;
    norm_vals[0] = 1.0 / 255.0f;
    norm_vals[1] = 1.0 / 255.0f;
    norm_vals[2] = 1.0 / 255.0f;

    return 0;
}

int YoloV8::detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    int width = rgb.cols;
    int height = rgb.rows;
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h) {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    } else {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolo.create_extractor();
    ex.input("in0", in_pad);

    std::vector<Object> proposals;
    ncnn::Mat out;
    ex.extract("out0", out);

    std::vector<int> strides = {8, 16, 32};
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);
    generate_proposals(grid_strides, out, prob_threshold, proposals);
    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    int count = picked.size();
    objects.resize(count);

    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    std::sort(objects.begin(), objects.end(), [](const Object& a, const Object& b) {
        return a.rect.area() > b.rect.area();
    });

    return 0;
}

int YoloV8::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        cv::rectangle(rgb, obj.rect, cv::Scalar(255, 0, 0));
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);
        cv::putText(rgb, text, cv::Point(x, y + label_size.height),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }

    return 0;
}

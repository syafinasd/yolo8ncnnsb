#include "yoloV8.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

const char* class_names[] = {
    'Adem Sari Panas Dalam', 'Ahh Rasa Roti Goguma Korea', 'Azarine Acne Gentle Cleansing Foam',
    'Azarine Sunscreen Gel', 'Barakat Pasta Gigi', 'Biore Pore Pack with Charcoal', 'Biore UV', 
    'Cleo Air Mineral', 'Club Air Mineral', 'Eiem Beauty Natural Sunscreen', 
    'Emeron Lovely White UV Jeju Orange', 'Emina Bright Stuff Face Wash', 'Emina Sun Protection', 
    'Energen Rasa Coklat', 'Eternalplus', 'Focallure 3 in 1 Palette', 'FreshCare Hot', 
    'Gimbori Rumput Laut Kering Tabur', 'Hot In Cream', 'Implora Jelly Tint', 
    'Indomie Rasa Kaldu Ayam', 'Indomilk Korean Black Latte', 'Le Minerale', 
    'Makarizo Hair Energy Morning Dew', 'Master Potato Crisps Rendang', 
    'Milku Susu UHT Rasa Coklat Premium', 'Milku Susu UHT Rasa Stroberi', 'Mixagrip Flu - Batuk', 
    'Nano Nano Milky Stroberi', 'Neo Coffee', 'Pop Ice Rasa Anggur', 'Pop Ice Rasa Cokelat', 
    'Pop Ice Rasa Taro', 'Prochiz Spready Keju Oles', 'Promag Herbal', 'Qlife Menstrual Care', 
    'Richeese Pasta Keju', 'Skintific Sunscreen Mist', 'Soffell Wangi Apel', 'Standart Pen AE7', 
    'Teh Javana', 'Tolakangin', 'Tolakangin Anak', 'Ultramilk Rasa Taro', 'Vape Mat 4.1 MV', 
    'Vit Air Mineral', 'Vital Ear Oil', 'Vitamin B', 'Yplogy Hair Mask Gingseng', 'Zen Antibacterial'
};

// Helper function for fast exponential calculation
static float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

// Sigmoid function implementation
static float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

// Calculates intersection area between two bounding boxes
static float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

// In-place quicksort for descending order based on object probabilities
static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    if (left < j) qsort_descent_inplace(objects, left, j);
    if (i < right) qsort_descent_inplace(objects, i, right);
}

// Wrapper function for descending order quicksort
static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

// Non-Maximum Suppression (NMS) to filter overlapping detections
static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = objects.size();
    std::vector<float> areas(n);

    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.width * objects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];

        bool keep = true;
        for (size_t j = 0; j < picked.size(); j++)
        {
            const Object& b = objects[picked[j]];

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            float IoU = inter_area / union_area;

            if (IoU > nms_threshold)
            {
                keep = false;
                break;
            }
        }

        if (keep)
        {
            picked.push_back(i);
        }
    }
}

// Generates grid coordinates and stride for YOLO-like models
static void generate_grids_and_stride(const int target_w, const int target_h, std::vector<int>& strides, std::vector<GridAndStride>& grid_strides)
{
    for (int i = 0; i < (int)strides.size(); i++)
    {
        int stride = strides[i];
        int num_grid_w = target_w / stride;
        int num_grid_h = target_h / stride;
        for (int g1 = 0; g1 < num_grid_h; g1++)
        {
            for (int g0 = 0; g0 < num_grid_w; g0++)
            {
                GridAndStride gs;
                gs.grid0 = g0;
                gs.grid1 = g1;
                gs.stride = stride;
                grid_strides.push_back(gs);
            }
        }
    }
}

// Generates object proposals from network predictions
static void generate_proposals(std::vector<GridAndStride> grid_strides, const ncnn::Mat& pred, float prob_threshold, std::vector<Object>& objects)
{
    const int num_points = grid_strides.size();
    const int num_class = 50; // Number of classes, adjust according to your model
    const int reg_max_1 = 16;

    for (int i = 0; i < num_points; i++)
    {
        const float* scores = pred.row(i) + 4 * reg_max_1;

        int label = -1;
        float score = -FLT_MAX;
        for (int k = 0; k < num_class; k++)
        {
            float confidence = scores[k];
            if (confidence > score)
            {
                label = k;
                score = confidence;
            }
        }
        float box_prob = sigmoid(score);

        if (box_prob >= prob_threshold)
        {
            // Process bounding box predictions
            ncnn::Mat bbox_pred(reg_max_1, 4, (void*)pred.row(i));

            // Softmax layer processing (if needed)
            ncnn::Layer* softmax = ncnn::create_layer("Softmax");
            ncnn::ParamDict pd;
            pd.set(0, 1); // axis
            pd.set(1, 1);
            softmax->load_param(pd);

            ncnn::Option opt;
            opt.num_threads = 1;
            opt.use_packing_layout = false;

            softmax->create_pipeline(opt);
            softmax->forward_inplace(bbox_pred, opt);
            softmax->destroy_pipeline(opt);
            delete softmax;

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++)
            {
                float dis = 0.f;
                const float* dis_after_sm = bbox_pred.row(k);
                for (int l = 0; l < reg_max_1; l++)
                {
                    dis += l * dis_after_sm[l];
                }

                pred_ltrb[k] = dis * grid_strides[i].stride;
            }

            float pb_cx = (grid_strides[i].grid0 + 0.5f) * grid_strides[i].stride;
            float pb_cy = (grid_strides[i].grid1 + 0.5f) * grid_strides[i].stride;

            float x0 = pb_cx - pred_ltrb[0];
            float y0 = pb_cy - pred_ltrb[1];
            float x1 = pb_cx + pred_ltrb[2];
            float y1 = pb_cy + pred_ltrb[3];

            Object obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = label;
            obj.prob = box_prob;

            objects.push_back(obj);
        }
    }
}

// Constructor
YoloV8::YoloV8()
{
    // Constructor implementation, currently empty
}

// Loads the YOLOv8 model
int YoloV8::load(int _target_size)
{
    // Clear existing model
    yolo.clear();

    // Set ncnn options
    yolo.opt = ncnn::Option();
    yolo.opt.num_threads = 4; // Number of threads for inference

    // Load model parameters and binary
    yolo.load_param("./model.ncnn.param");
    yolo.load_model("./model.ncnn.bin");

    // Set target size and normalization values
    target_size = _target_size;
    mean_vals[0] = 103.53f;
    mean_vals[1] = 116.28f;
    mean_vals[2] = 123.675f;
    norm_vals[0] = 1.0 / 255.0f;
    norm_vals[1] = 1.0 / 255.0f;
    norm_vals[2] = 1.0 / 255.0f;

    return 0;
}

// Performs object detection on input image
int YoloV8::detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold, float nms_threshold)
{
    int width = rgb.cols;
    int height = rgb.rows;

    // Resize and pad input image
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);

    // Pad to target_size
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    // Subtract mean and normalize
    in_pad.substract_mean_normalize(0, norm_vals);

    // Create ncnn extractor and set input
    ncnn::Extractor ex = yolo.create_extractor();
    ex.input("in0", in_pad);

    // Extract output
    std::vector<Object> proposals;
    ncnn::Mat out;
    ex.extract("out0", out);

    // Define strides and generate grids
    std::vector<int> strides = {8, 16, 32}; // Possible strides, adjust according to model
    std::vector<GridAndStride> grid_strides;
    generate_grids_and_stride(in_pad.w, in_pad.h, strides, grid_strides);

    // Generate object proposals
    generate_proposals(grid_strides, out, prob_threshold, proposals);

    // Sort proposals by score
    qsort_descent_inplace(proposals);

    // Apply NMS
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    // Adjust bounding boxes and scale back to original size
    int count = picked.size();
    objects.resize(count);

    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // Adjust to original image size
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // Clip to image boundaries
        x0 = std::max(std::min(x0, (float)(width - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(height - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(width - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(height - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    // Sort objects by area
    std::sort(objects.begin(), objects.end(), [](const Object& a, const Object& b) {
        return a.rect.area() > b.rect.area();
    });

    return 0;
}

// Draws bounding boxes and labels on input image
int YoloV8::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    // Draw bounding boxes and labels for detected objects
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

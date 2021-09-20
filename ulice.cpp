#include <iostream>
#include <sstream>
#include <fstream>
#include <numeric> 

#include "opencv2\opencv.hpp"


using namespace cv;
using namespace dnn;
using namespace std;



vector<Point2f> punkty_na_obrazie;
vector<Point2f> H_punkty_klikniete;
vector<Point2f> H_punkty_przeliczone;
Mat obraz_z_kamery;

void klikniecie_mysza(int event, int x, int y, int, void *parametr) {
	if (event == EVENT_LBUTTONDOWN)
	{
		//klikniety punkt wpisywany jest do wektora punktow
		Point p(x, y);
		punkty_na_obrazie.push_back(p);
		cout << p << endl;

		if ((bool)parametr == true)
		{
			circle(obraz_z_kamery, p, 5, Scalar(0, 0, 255));
			imshow("obraz", obraz_z_kamery);
			waitKey(1);
		}
	}
}

void klikniecie_mysza_z_homografia(int event, int x, int y, int, void *parametr) {
	//przelicza wspolrzedne kliknietego punktu na obrazie oryginalnym 
	//na wspolrzedne na docelowej plaszczyznie
	Mat* H = (Mat*)parametr;
	if (event == EVENT_LBUTTONDOWN)
	{
		// klikniety punkt
		Point3d p1(x, y, 1);
		Point3d p2 = Point3d(Mat(*H*Mat(p1))); //to jest po prostu H*p1, ale w OpenCV trzeba pomanipulowa� typami danych
		p2 /= p2.z;
		H_punkty_klikniete.push_back(Point2f(p1.x, p1.y));
		H_punkty_przeliczone.push_back(Point2f(p2.x, p2.y));

		cout << "Wsp. na obrazie: " << Point2f(p1.x, p1.y) << " wzg. pasow: " << Point2f(p2.x, p2.y) << endl;


	}
}


class Yolo
{
public:
	vector<string> class_names;
	float confidence_threshold;
	float nms_threshold;

	vector<vector<Rect>> valid_boxes;
	vector <vector<float>> valid_scores;
	float inference_fps, total_fps;

private:
	int num_classes;
	dnn::Net net;
	vector<Mat> detections;
	vector<vector<int>> indices;
	vector < vector<Rect>> boxes;
	vector < vector<float>> scores;
	vector<string> output_names;
	Mat blob;



	// colors for bounding boxes
	vector <Scalar> colors = {
		{0, 0, 255},
		{0, 255, 0},
		{55, 55, 255},
		{255, 255, 0},
		{0, 255, 255},
		{ 255,0, 255}
	};
	const int num_colors = colors.size();



public:
	Yolo(string yolo_cfg, string yolo_weights, string classes_filename, double confidence_threshold_ = .4, double nms_threshold_ = .4)
	{
		{
			ifstream class_file(classes_filename);
			if (!class_file)
			{
				cerr << "failed to open classes.txt\n";
				exit;
			}

			string line;
			while (getline(class_file, line))
				class_names.push_back(line);
			num_classes = class_names.size();
		}

		confidence_threshold = confidence_threshold_;
		nms_threshold = nms_threshold_;

		net = dnn::readNetFromDarknet(yolo_cfg, yolo_weights);
		//net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
		//net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
		output_names = net.getUnconnectedOutLayersNames();

		boxes.resize(num_classes);
		valid_boxes.resize(num_classes);
		indices.resize(num_classes);
		scores.resize(num_classes);
		valid_scores.resize(num_classes);

	}

	void detect(Mat &frame, Size size = Size(416, 416))
	{
		auto total_start = chrono::steady_clock::now();
		dnn::blobFromImage(frame, blob, 0.00392, size, Scalar(), true, false, CV_32F);
		net.setInput(blob);

		auto dnn_start = chrono::steady_clock::now();
		net.forward(detections, output_names);
		auto dnn_end = chrono::steady_clock::now();

		for (int c = 0; c < num_classes; c++)
		{
			boxes[c].clear();
			scores[c].clear();
			indices[c].clear();
		}
		for (auto& output : detections)
		{
			const auto num_boxes = output.rows;
			for (int i = 0; i < num_boxes; i++)
			{
				auto x = output.at<float>(i, 0) * frame.cols;
				auto y = output.at<float>(i, 1) * frame.rows;
				auto width = output.at<float>(i, 2) * frame.cols;
				auto height = output.at<float>(i, 3) * frame.rows;
				Rect rect(x - width / 2, y - height / 2, width, height);

				for (int c = 0; c < num_classes; c++)
				{
					auto confidence = *output.ptr<float>(i, 5 + c);
					if (confidence >= confidence_threshold)
					{
						boxes[c].push_back(rect);
						scores[c].push_back(confidence);
					}
				}
			}
		}

		for (int c = 0; c < num_classes; c++)
		{
			dnn::NMSBoxes(boxes[c], scores[c], 0.0, nms_threshold, indices[c]);
			//uporzadkowanie boxes i scores by zawieraly tylko to co trzeba a nie wszystko
			valid_boxes[c].clear();
			valid_scores[c].clear();
			for (int i = 0; i < indices[c].size(); i++)
			{
				valid_boxes[c].push_back(boxes[c][indices[c][i]]);
				valid_scores[c].push_back(scores[c][indices[c][i]]);
			}

		}


		auto total_end = std::chrono::steady_clock::now();

		inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
		total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();

	}

	void draw_detections(Mat &frame, vector<int> class_nums_to_draw_ = {}, bool draw_labels_ = false)
	{
		if (class_nums_to_draw_.empty())

			class_nums_to_draw_.resize(num_classes);
		iota(begin(class_nums_to_draw_), end(class_nums_to_draw_), 0);


		for (auto const& c : class_nums_to_draw_)
		{
			for (size_t i = 0; i < valid_boxes[c].size(); ++i)
			{
				const auto color = colors[c % num_colors];

				const auto& rect = valid_boxes[c][i];
				rectangle(frame, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), color, 1);

				if (draw_labels_)
				{
					std::ostringstream label_ss;
					label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << valid_scores[c][i];
					auto label = label_ss.str();


					int baseline;
					auto label_bg_sz = getTextSize(label.c_str(), FONT_HERSHEY_COMPLEX_SMALL, .8, 1, &baseline);
					//rectangle(frame, Point(rect.x, rect.y - label_bg_sz.height - baseline - 5), Point(rect.x + label_bg_sz.width, rect.y), color, FILLED);
					//putText(frame, label.c_str(), Point(rect.x, rect.y - baseline), FONT_HERSHEY_COMPLEX_SMALL, .8, Scalar(0, 0, 0));
				}
			}
		}
	}



};

void draw_10m_lines(int y1, Mat H1) {
	Point3d p1(-5, y1, 1);
	Point3d p2(5, y1, 1);
	Point3d p1b = Point3d(Mat((H1.inv()) * Mat(p1)));
	Point3d p2b = Point3d(Mat((H1.inv()) * Mat(p2)));

	p1b /= p1b.z;
	p2b /= p2b.z;

	line(obraz_z_kamery, Point(p1b.x, p1b.y), Point(p2b.x, p2b.y), Scalar(0,0,255), 1);
}

void draw_10m_lines_straight(int y1, Mat obraz2, Mat H, Mat H1) {
	Point3d p1(-5, y1, 1);
	Point3d p2(5, y1, 1);
	Point3d p1b = Point3d(Mat((H1.inv()) * Mat(p1)));
	Point3d p2b = Point3d(Mat((H1.inv()) * Mat(p2)));
	Point3d p1c = Point3d(Mat((H) * Mat(p1b)));
	Point3d p2c = Point3d(Mat((H) * Mat(p2b)));

	p1c /= p1c.z;
	p2c /= p2c.z;

	line(obraz2, Point(p1c.x, p1c.y), Point(p2c.x, p2c.y), Scalar(0,255,0), 10); //255
}

void draw_line_object(Point br, int width, Mat obraz2, Mat H, Mat H1) {
	Point3d p1(br.x, br.y, 1);
	Point3d p2(br.x - width, br.y, 1);
	Point3d p1b = Point3d(Mat((H) * Mat(p1)));
	Point3d p2b = Point3d(Mat((H) * Mat(p2)));

	p1b /= p1b.z;
	p2b /= p2b.z;

	line(obraz2, Point(p1b.x, p1b.y), Point(p2b.x, p2b.y), Scalar(0,0,255), 10);
}

float Calc_dist(Point rog, Mat H1) {
	
	Point3d p1(rog.x, rog.y, 1);
	Point3d p1b = Point3d(Mat((H1) * Mat(p1)));
	p1b /= p1b.z;
	float dist = p1b.y;
	
	return dist;
}

int ulice()
{
	//wykorzystanie homografii w scenie ulicznej

	VideoCapture kamera("ulice4.mp4");
	kamera >> obraz_z_kamery;
	if (obraz_z_kamery.empty()) return 1; //jak film si� nie za�aduje, program od razu zako�czy dzia�anie

	Mat kopia = obraz_z_kamery.clone(); //kopia obrazu z kamery

	namedWindow("obraz", 0); //skalowalne okienko
	putText(obraz_z_kamery, String("Oznacz 4 punkty na plaszczyznie"), Point(0, 45), 0, .8, CV_RGB(255, 0, 0), 2);
	imshow("obraz", obraz_z_kamery);
	waitKey(1);
	setMouseCallback("obraz", klikniecie_mysza, (void*)true); //ustawienie funkcji czytajacej mysz na danym oknie z obrazem (true oznacza, ze rysowane sa koleczka w miejscu klikniecia)

	while (punkty_na_obrazie.size() < 4) waitKey(1); //poczekanie na 4 punkty_na_obrazie

	//dok�adniejsze wykrycie naro�nik�w - oryginalnie stosowane przy detekcji szachownicy - nie musimy bardzo precyzyjnie klika� (ale bez przesady
	Mat tmp;
	cvtColor(obraz_z_kamery, tmp, COLOR_BGR2GRAY);
	cornerSubPix(tmp, punkty_na_obrazie, Size(20, 20), Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 1000, 0.01));


	//narysowanie wielokata laczacego punkty_na_obrazie dla weryfikacji
	for (int i = 0; i < 4; i++)
		line(obraz_z_kamery, punkty_na_obrazie[i], punkty_na_obrazie[(i + 1) % 4], CV_RGB(255, 100, 100), 1, LineTypes::LINE_AA);

	imshow("obraz", obraz_z_kamery);
	waitKey(1);

	//wygenerowanie lokalnych wspolrzednych X,Y zaznaczonych naroznikow
	//punkty_na_obrazie byly zaznaczane w takiej kolejnosci, by odpowiadaly ponizszym wspolrzednym punktow na obiekcie

	float s = 1, x = 500, y = 5000; //przesuni�cie pocz�tku uk�adu wsp�rz�dnych 300;5000
	vector<Point2f>  punkty_na_obiekcie{ { 0 + x, 0 + y },{ s * 350 + x,+y },{ s * 350 + x, s * 400 + y },{ 0 + x, s * 400 + y } }; //pasy
	Mat H = findHomography(punkty_na_obrazie, punkty_na_obiekcie);

	vector<Point2f>  punkty_wzgledem_samochodu{ {-1.5,10 },{ 2,10},{ 2,6},{-1.5,6} }; // okre�lenie odleg�o�ci wzgl�dem samochodu
	Mat H1 = findHomography(punkty_na_obrazie, punkty_wzgledem_samochodu);



	Mat obraz2;
	warpPerspective(kopia, obraz2, H, Size(1000+x, 500+y));		//350;400


	for (int num_lines = 1; num_lines < 6; num_lines++) {
		draw_10m_lines_straight(num_lines * 10, obraz2, H, H1);
	}

	namedWindow("wyprostowany", 0);
	imshow("wyprostowany", obraz2);
	setMouseCallback("wyprostowany", klikniecie_mysza);
	setMouseCallback("obraz", klikniecie_mysza_z_homografia, (void*)&H1);

	//za�adowanie sieci neuronowej
	Yolo yolo("yolov4-tiny.cfg", "yolov4-tiny.weights", "classes.txt", .4, .2);
	yolo.detect(obraz_z_kamery);
	yolo.draw_detections(obraz_z_kamery, {}, true);

	
	for (int num_lines = 1; num_lines < 6; num_lines++) {
		draw_10m_lines(num_lines * 10, H1);
	}

	imshow("obraz", obraz_z_kamery);
	waitKey(0);

	//nowy punkt -1.5,10


	int k = 0;
	while (k != 27)
	{
		//for(int i=0;i<5;i++) //je�eli wolno chodzi, mo�na odczytywa� co kt�r�� klatk�
		kamera >> obraz_z_kamery;
		if (obraz_z_kamery.empty()) return 1; //jak film si� sko�czy, program zako�czy dzia�anie
		Mat kopia = obraz_z_kamery.clone();
		warpPerspective(kopia, obraz2, H, Size(1000 + x, 500 + y));

		//widok z lotu ptaka - linie
		for (int num_lines = 1; num_lines < 6; num_lines++) {
			draw_10m_lines_straight(num_lines * 10, obraz2, H, H1);
		}

		//widok zwyk�y - linie
		for (int num_lines = 1; num_lines <= 6; num_lines++) {
			draw_10m_lines(num_lines * 10, H1);
		}

		yolo.detect(obraz_z_kamery, Size(416, 416)); //im wi�ksza liczba tym dok�adniejsza lecz wolniejsza detekcja obiekt�w
		yolo.draw_detections(obraz_z_kamery, {}, true);
		
		for (int klasa : {0, 1, 2, 3}) //numery interesuj�cych nas klas z pliku classes.txt
			for (Rect wykryty_obiekt : yolo.valid_boxes[klasa]) //wykryty obiekt jest oznaczany prostok�tem
			{
				//prostok�t ma m.in. parametry tl - punkt lewy g�rny, br - prawy dolny, a wi�c ich �rednia to �rodek prostok�ta
				//inny, by� mo�e przydatny, parametr to width - szeroko��
				Point srodek = (wykryty_obiekt.tl() + wykryty_obiekt.br()) / 2;
				float odleglosc = Calc_dist(wykryty_obiekt.br(), H1);
				putText(obraz_z_kamery, format("%.0f", odleglosc), srodek, 0, .8, CV_RGB(0, 255, 0), 2);
				draw_line_object(wykryty_obiekt.br(), wykryty_obiekt.width, obraz2, H, H1);
			}

		imshow("wyprostowany", obraz2);
		imshow("obraz", obraz_z_kamery);
		k = waitKey(1);
		if (k == ' ') waitKey(0);
	}

	//do zrobienia:
	//1. Zmodyfikowa� parametry homografii by by�o wida� ulic� od samochodu do 50 metr�w na prz�d od klikni�tego naro�nika pas�w oraz po 5 metr�w w lewo i prawo
	//od osi samochodu
	//2. Sprawdzi� czy wsp�rz�dne po klikni�ciu w obraz s� prawid�owo przeliczane i czy rzut z lotu ptaka jest poprawny
	//3. Uzupe�ni� transformacj� obrazu w p�tli while, tak aby ca�y czas by�o wida� rzut z g�ry
	//4. Poeksperymentowa� z rozdzielczo�ci� i rodzajem (wersja pe�na bez "-tiny")
	//5. Jak wyznaczy� odleg�o�� od wykrytego pojazdu - zak�adamy, �e lewy dalszy r�g pas�w ma wsp�rz�dne -1,5x10 metr�w wzgl�dem kamery
	//6. Wyznaczy� drug� homografi� H2
	//7. Maj�c H2 narysowa� linie co 10 metr�w przed samochodem na obrazie z kamery - jak przeliczy� ich wsp�rz�dne ze �wiata na obrazowe?
	//8. Takie same linie chcemy widzie� na obrazie z lotu ptaka - jak to przeliczy�?
	//9. Jak wyznaczy� odleg�o�� od wykrytych obiekt�w? Jaki warunek musi by� spe�niony?
	//10. Efekt ko�cowy: na �rodku ka�dego pojazdu oraz przechodnia pojawia si� liczba oznaczaj�ca odleg�o�� wzd�u� kierunku jazdy pomi�dzy naszym
	//samochodem, a tym�e obiektem; z dok�adno�ci� do 1 m. Nie wy�wietlamy odleg�o�ci powy�ej 50 m.

	return 0;
}





int main()
{

	ulice();


	return 0;
}



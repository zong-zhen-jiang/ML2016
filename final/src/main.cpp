// g++ main.cpp -std=c++11 -lpthread -O3
#include <assert.h>
#include <ctime>
#include <fstream>
#include <map>
#include <set>
#include <stdio.h>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "csv.h"
#include "ftrl.hpp"


using namespace std;


const string DATA_DIR = "../data";


int cmp(const pair<uint, double> & x, const pair<uint, double> & y)
{
    return x.second > y.second;
}
void predict(Ftrl & ftrl,
        const string & oriInCsvPath,
        const string & inCsvPath,
        const string & outCsvPath,
        int fuck);
void train(Ftrl & ftrl,
        const string & trainFilePath,
        int fuck0=0,
        int fuck1=0,
        int fuck2=0);


int main()
{
    Ftrl ftrl(0.0130, 0.0, 0.0, 0.0, pow(2., 26.));
    int nbEpoch = 100;
    for (int i = 0; i < nbEpoch; i++) {
        train(ftrl, DATA_DIR + "/_tr_sp.csv", i + 1, nbEpoch, 67237323);
        // train(ftrl, DATA_DIR + "/_tr.csv", i + 1, nbEpoch, 87141733);
    }
    predict(ftrl,
            DATA_DIR + "/_clicks_valid_sp.csv",
            DATA_DIR + "/_va_sp.csv",
            DATA_DIR + "/_va_out.csv",
            19904410);
    predict(ftrl,
            DATA_DIR + "/clicks_test.csv",
            DATA_DIR + "/_te.csv",
            DATA_DIR + "/_te_out.csv",
            32225163);

    return 0;
}


void predict(Ftrl & ftrl,
        const string & oriInCsvPath,
        const string & inCsvPath,
        const string & outCsvPath,
        int fuck)
{
    //
    vector<pair<uint, double> > clickProbVec;
    uint prevDispId = 0;
    //
    io::CSVReader<3> oriInCsv(oriInCsvPath);
    oriInCsv.read_header(io::ignore_missing_column,
            "display_id",
            "ad_id",
            "clicked");
    uint dispId = 0;
    uint adId = 0;
    uint clickedOri = 0;
    //
    io::CSVReader<18> inCsv(inCsvPath);
    inCsv.read_header(io::ignore_no_column,
            "clicked",
            "disp",
            "ad",
            "ad_doc",
            "camp",
            "ader",
            "uuid",
            "plat",
            "loc",
            "loc_c",
            "loc_s",
            "loc_d",
            "ad_doc_src",
            "ad_doc_puber",
            "ad_doc_cat",
            "ad_doc_top",
            "nb_ads",
            "leak");
    uint indices[18] = {0};
    uint clicked = 0;
    uint disp = 0;
    uint ad = 0;
    uint adDoc = 0;
    uint camp = 0;
    uint ader = 0;
    uint uuid = 0;
    uint plat = 0;
    uint loc = 0;
    uint locC = 0;
    uint locS = 0;
    uint locD = 0;
    uint adDocSrc = 0;
    uint adDocPuber = 0;
    uint adDocCat = 0;
    uint adDocTop = 0;
    uint nbAds = 0;
    uint leak = 0;
    //
    fstream outCsv;
    outCsv.open(outCsvPath, ios::out);
    outCsv << "display_id,ad_id\n";
    //
    clock_t t1 = clock();
    int cnt = 0;
    while (inCsv.read_row(
            clicked,
            disp,
            ad,
            adDoc,
            camp,
            ader,
            uuid,
            plat,
            loc,
            locC,
            locS,
            locD,
            adDocSrc,
            adDocPuber,
            adDocCat,
            adDocTop,
            nbAds,
            leak)) {

        if (cnt++ % 15566 == 0) {
            printf("\rPredicting... (%8d/%d), %.1f secs",
                    cnt, fuck, (clock() - t1) / (float) CLOCKS_PER_SEC);
            fflush(stdout);
        }

        //
        oriInCsv.read_row(dispId, adId, clickedOri);

        //
        indices[0] = 0;
        //
        indices[1] = disp;
        indices[2] = ad;
        //
        indices[3] = adDoc;
        indices[4] = camp;
        indices[5] = ader;
        //
        indices[6] = uuid;
        indices[7] = plat;
        indices[8] = loc;
        indices[9] = locC;
        indices[10] = locS;
        indices[11] = locD;
        //
        indices[12] = adDocSrc;
        indices[13] = adDocPuber;
        //
        indices[14] = adDocCat;
        indices[15] = adDocTop;
        //
        indices[16] = nbAds;
        indices[17] = leak;

        //
        double p = ftrl.predict(indices, 18);

        if (clickProbVec.size() == 0) {
            // Will only run once.
            clickProbVec.push_back(make_pair(adId, p));
            prevDispId = dispId;
            continue;
        }

        if (prevDispId != dispId) {
            sort(clickProbVec.begin(), clickProbVec.end(), cmp);
            outCsv << prevDispId << ",";
            for (int i = 0; i < clickProbVec.size(); i++) {
                outCsv << clickProbVec[i].first;
                if (i != clickProbVec.size() - 1) {
                    outCsv << " ";
                }
            }
            outCsv << "\n";
            clickProbVec = vector<pair<uint, double> >();
        }

        //
        clickProbVec.push_back(make_pair(adId, p));
        prevDispId = dispId;
    }
    printf("\n");

    sort(clickProbVec.begin(), clickProbVec.end(), cmp);
    outCsv << prevDispId << ",";
    for (int i = 0; i < clickProbVec.size(); i++) {
        outCsv << clickProbVec[i].first;
        if (i != clickProbVec.size() - 1) {
            outCsv << " ";
        }
    }
    outCsv << "\n";
    outCsv.close();
}

void train(Ftrl & ftrl,
        const string & trainFilePath,
        int fuck0,
        int fuck1,
        int fuck2)
{
    //
    io::CSVReader<18> in(trainFilePath);
    in.read_header(io::ignore_missing_column,
            "clicked",
            "disp",
            "ad",
            "ad_doc",
            "camp",
            "ader",
            "uuid",
            "plat",
            "loc",
            "loc_c",
            "loc_s",
            "loc_d",
            "ad_doc_src",
            "ad_doc_puber",
            "ad_doc_cat",
            "ad_doc_top",
            "nb_ads",
            "leak");
    uint indices[18] = {0};
    uint clicked = 0;
    uint disp = 0;
    uint ad = 0;
    uint adDoc = 0;
    uint camp = 0;
    uint ader = 0;
    uint uuid = 0;
    uint plat = 0;
    uint loc = 0;
    uint locC = 0;
    uint locS = 0;
    uint locD = 0;
    uint adDocSrc = 0;
    uint adDocPuber = 0;
    uint adDocCat = 0;
    uint adDocTop = 0;
    uint nbAds = 0;
    uint leak = 0;
    clock_t t1 = clock();
    int cnt = 0;
    while (in.read_row(
            clicked,
            disp,
            ad,
            adDoc,
            camp,
            ader,
            uuid,
            plat,
            loc,
            locC,
            locS,
            locD,
            adDocSrc,
            adDocPuber,
            adDocCat,
            adDocTop,
            nbAds,
            leak)) {

        if (cnt++ % 15566 == 0) {
            printf("\r[Epoch %d/%d] %d/%d, %.1f secs",
                    fuck0, fuck1, cnt, fuck2,
                    (clock() - t1) / (float) CLOCKS_PER_SEC);
            fflush(stdout);
        }

        assert(clicked != 87);
        double y = clicked;

        //
        indices[0] = 0;
        //
        indices[1] = disp;
        indices[2] = ad;
        //
        indices[3] = adDoc;
        indices[4] = camp;
        indices[5] = ader;
        //
        indices[6] = uuid;
        indices[7] = plat;
        indices[8] = loc;
        indices[9] = locC;
        indices[10] = locS;
        indices[11] = locD;
        //
        indices[12] = adDocSrc;
        indices[13] = adDocPuber;
        //
        indices[14] = adDocCat;
        indices[15] = adDocTop;
        //
        indices[16] = nbAds;
        indices[17] = leak;

        //
        double p = ftrl.predict(indices, 18);
        ftrl.update(indices, 18, p, y);
    }
    printf("\n");
}

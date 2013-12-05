#include <iostream>
#include <string>

using namespace std;

int main_opt();
int main_2ff();
int main_2cost();
int main_2bp();

int main(int argc, const char *argv[])
{
    if (argc < 2)
	{
        cout << "unitTests selector[opt, 2ff, 2cost, 2bp]" << endl;
		return -1;
	}

    string selector(argv[1]);
    cout << selector << endl;

    if (selector == "opt")
        main_opt();
    if (selector == "2ff")
        main_2ff();
    if (selector == "2cost")
        main_2cost();
    if (selector == "2bp")
        main_2bp();

    return 0;
}

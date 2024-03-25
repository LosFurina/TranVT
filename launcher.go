package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"
)

//--Dataset -d      Dataset from ['swat', 'wadi']
//--Model -m        Model name from [TranAD, TranVT]
//--Lr -l           learning rate in training
//--win_size -ws    windows size in splitting Dataset to type of time series
//--batch_size -bs  batch size
//--Epochs -e       epoch times
//--Test            Test Model
//--exp_id -ei      tested checkpoint id
//--top_k -t        the top k score used in evaluation algorithm

func main() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Println("Welcome to TranVT Anomaly detection:")
	fmt.Println("We will set some arguments here if you are not familiar with this project:")
	fmt.Println("Press enter to continue:")

	_, _ = reader.ReadString('\n')

	fmt.Println("Continuing...")

	args := new(Args)
	args.Set()
	cmd := args.GetCmd()
	fmt.Println(cmd)

	// 将命令写入 bat 文件
	if err := writeCmdToBat(cmd, "launcher.bat"); err != nil {
		fmt.Println("Error writing command to bat file:", err)
	} else {
		fmt.Println("Command has been written to command.bat successfully.")
	}
}

func writeCmdToBat(cmd, filename string) error {
	// 创建或打开 bat 文件
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// 将命令写入文件
	_, err = file.WriteString(cmd)
	if err != nil {
		return err
	}

	return nil
}

type Args struct {
	Dataset   string
	Model     string
	Lr        string
	WinSize   string
	BatchSize string
	Epochs    string
	Test      string
	ExpId     string
	TopK      string
}

func (a *Args) Set() {
	reader := bufio.NewReader(os.Stdin)

	fmt.Print("Train or Test [y/n]: ")
	op, _ := reader.ReadString('\n')
	op = strings.TrimSpace(op)

	fmt.Println("Dataset: ")
	fmt.Println("1.swat: ")
	fmt.Println("2.wadi: ")
	fmt.Scanln(&a.Dataset)
	if a.Dataset == "1" {
		a.Dataset = "swat"
	} else if a.Dataset == "2" {
		a.Dataset = "wadi"
	} else {
		panic("Invalid input")
	}
	fmt.Print("Model: ")
	fmt.Print("1.TranVT: ")
	fmt.Print("2.TranVTP: ")
	fmt.Print("3.TranVTS: ")
	fmt.Scanln(&a.Model)
	if a.Model == "1" {
		a.Model = "TranVT"
	} else if a.Model == "2" {
		a.Model = "TranVTP"
	} else if a.Model == "3" {
		a.Model = "TranVTS"
	} else {
		panic("Invalid input")
	}
	if op == "y" || op == "Y" {
		fmt.Print("Lr: ")
		fmt.Scanln(&a.Lr)

		fmt.Print("Windows size: ")
		fmt.Scanln(&a.WinSize)

		fmt.Print("Batch size: ")
		fmt.Scanln(&a.BatchSize)

		fmt.Print("Epochs: ")
		fmt.Scanln(&a.Epochs)

		a.Test = "y"
	} else if op == "n" || op == "N" {
		fmt.Println("Experiment id: ")
		expDirs, _ := os.ReadDir("./exp")
		for i, expDir := range expDirs {
			if expDir.IsDir() {
				fmt.Printf("%d: %s\n", i+1, expDir.Name())
			}
		}
		var choice int
		fmt.Scanln(&choice)
		var expID string
		for i, expDir := range expDirs {
			if expDir.IsDir() {
				if i+1 == choice {
					expID = expDir.Name()
					break
				}
			}
		}
		a.ExpId = expID
		fmt.Print("Top k: ")
		fmt.Scanln(&a.TopK)

		a.Test = "n"
	} else {
		fmt.Println("Your input is invalid!")
		os.Exit(1)
	}
}

func (a *Args) GetCmd() string {
	var cmd string
	if a.Test == "y" || a.Test == "Y" {
		cmd = fmt.Sprintf("python ./train.py --dataset %s --model %s --lr %s --win_size %s --batch_size %s --epochs %s",
			a.Dataset, a.Model, a.Lr, a.WinSize, a.BatchSize, a.Epochs)
	} else if a.Test == "n" || a.Test == "N" {
		cmd = fmt.Sprintf("python ./test.py --dataset %s --model %s --test --exp_id %s --top_k %s",
			a.Dataset, a.Model, a.ExpId, a.TopK)
	} else {
		fmt.Println("Your test option is invalid!")
		os.Exit(1)
	}
	return cmd
}
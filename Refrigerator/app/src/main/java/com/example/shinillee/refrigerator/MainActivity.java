package com.example.shinillee.refrigerator;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.support.v7.widget.DividerItemDecoration;
import android.support.v7.widget.LinearLayoutManager;
import android.support.v7.widget.RecyclerView;
import android.util.Log;
import android.view.View;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ListView;

import com.example.shinillee.refrigerator.HttpConnection.GetRefItems;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.io.UnsupportedEncodingException;
import java.net.URLDecoder;
import java.util.ArrayList;
import java.util.Dictionary;
import java.util.concurrent.ExecutionException;

public class MainActivity extends AppCompatActivity{

    private String result;
    private ArrayList<RefItemInfo> mItems;
    private ListView listView;
    private ListviewAdapter adapter;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        mItems = new ArrayList<RefItemInfo>();

        listView = (ListView)findViewById(R.id.listview);
        adapter = new ListviewAdapter(this);

        View header = getLayoutInflater().inflate(R.layout.listview_header, null, false);
        listView.addHeaderView(header);


        setupView();


    }

    public void setupView(){
        //http 통신으로 공지사항 목록 받아오기
        GetRefItems getRefItems = new GetRefItems();
        try {
            result = getRefItems.execute().get();

            Gson gson = new GsonBuilder().disableHtmlEscaping().create();
            RefItemInfo[] refItemInfos = gson.fromJson(result, RefItemInfo[].class);

            for (RefItemInfo refItemInfo : refItemInfos) {
//                    String name = refItemInfo.getName();
//                    refItemInfo.setName(name);
//
//                    String inputDate = refItemInfo.getInputDate();
//                    refItemInfo.setWriter(writer);
//
//                    String content = refItemInfo.getContent();
//                    refItemInfo.setContent(content);
//                    mItems.add(refItemInfo);

                Log.d("@@@@test",""+refItemInfo.getIndex());
                Log.d("@@@@test",""+refItemInfo.getName());
                Log.d("@@@@test",""+refItemInfo.getInputDate());


                    adapter.add(refItemInfo);

            }
        } catch (InterruptedException e) {
            e.printStackTrace();
        } catch (ExecutionException e) {
            e.printStackTrace();
        }

        listView.setAdapter(adapter);
    }
}
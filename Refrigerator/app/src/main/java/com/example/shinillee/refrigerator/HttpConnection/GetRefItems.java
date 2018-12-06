package com.example.shinillee.refrigerator.HttpConnection;

import android.os.AsyncTask;
import android.util.Log;

import java.io.IOException;

import okhttp3.FormBody;
import okhttp3.OkHttpClient;
import okhttp3.Request;
import okhttp3.RequestBody;
import okhttp3.Response;

public class GetRefItems extends AsyncTask<Void, Void, String> {

    String answer;

    public GetRefItems() {

    }

    @Override
    protected String doInBackground(Void... params) {

        OkHttpClient client = new OkHttpClient();
        Response response;
        RequestBody requestBody = null;
        requestBody = new FormBody.Builder().build();
        Request request = new Request.Builder()
                .url("http://117.17.142.135:8080/RefrigeratorServer/getRefItems")
                .post(requestBody)
                .build();
        try {
            response = client.newCall(request).execute();
            answer = response.body().string();

        } catch (IOException e) {
            e.printStackTrace();
        }
        Log.d("answer", "" + answer);
        return answer;
    }

    @Override
    protected void onPostExecute(String s) {
        //super.onPostExecute(s);
        //doInBackground()로 부터 리턴된 값이 onPostExecute()의 매개변수로 넘어오므로 s를 출력한다.
    }

    public RequestBody selectStudentList(RequestBody requestBody) {
        return requestBody = new FormBody.Builder().build();
    }
}

package com.example.shinillee.refrigerator;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.DialogInterface;
import android.content.Intent;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.BaseAdapter;
import android.widget.TextView;
import android.widget.Toast;

import com.example.shinillee.refrigerator.HttpConnection.DeleteRefItemInfo;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;

public class ListviewAdapter extends BaseAdapter{

    private Context mContext;

    // 리스트 아이템 데이터를 저장할 배열
    private List<RefItemInfo> mItems;

    private ViewHolder holder = new ViewHolder();

    public ListviewAdapter(Context context){
        super();
        mContext = context;
        mItems = new ArrayList<RefItemInfo>();
    }

    @Override
    public int getCount() {
        return mItems.size();
    }

    @Override
    public Object getItem(int i) {
        return mItems.get(i);
    }

    @Override
    public long getItemId(int position) {
        return 0;
    }

    @Override
    public View getView(final int position, View view, ViewGroup viewGroup) {


        if(view == null) {
            LayoutInflater inflater = (LayoutInflater)viewGroup.getContext().getSystemService(Context.LAYOUT_INFLATER_SERVICE);
            view = inflater.inflate(R.layout.listview_item, viewGroup, false);

//            holder.index = (TextView)view.findViewById(R.id.index);
            holder.mName = (TextView)view.findViewById(R.id.name);
            holder.mInputDate = (TextView)view.findViewById(R.id.inputdate);
            holder.mExpiration = (TextView)view.findViewById(R.id.expiration);

            view.setTag(holder);

        } else {
            holder = (ViewHolder)view.getTag();
        }


//        holder.index.setText(String.valueOf(mItems.get(position).getIndex()));
        holder.mName.setText(String.valueOf(mItems.get(position).getName()));
        holder.mInputDate.setText(mItems.get(position).getInputDate());
        holder.mExpiration.setText(mItems.get(position).getExpiration());


        //공지사항 확인
        view.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Log.d("@@@@index",""+mItems.get(position).getIndex());

//                AlertDialog.Builder dialog = new AlertDialog.Builder(mContext);
//                dialog.setTitle("공지사항 내용");
//                TextView content = new TextView(mContext);
//                content.setText(mItems.get(position).getContent());
//
//                dialog.setView(content);
//
//
//                dialog.setNegativeButton("확인",new DialogInterface.OnClickListener() {
//                    public void onClick(DialogInterface dialog, int whichButton) {
//                    }
//                });
//                dialog.show();

            }
        });

        //공지사항 삭제
        view.setOnLongClickListener(new View.OnLongClickListener() {
            @Override
            public boolean onLongClick(View v) {

                AlertDialog.Builder dialog = new AlertDialog.Builder(mContext);
                dialog.setTitle("아이템 삭제");
                dialog.setMessage("정말 삭제 하시겠습니까?");
                dialog.setPositiveButton("확인",new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {

                        DeleteRefItemInfo deleteNoticeInfo = new DeleteRefItemInfo(String.valueOf(mItems.get(position).getIndex()));
                        try {
                            String result = deleteNoticeInfo.execute().get();
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        } catch (ExecutionException e) {
                            e.printStackTrace();
                        }

                        //리플레쉬
                        Intent intent = new Intent(mContext, MainActivity.class);
                        mContext.startActivity(intent);
                        ((Activity)mContext).finish();
                    }
                });
                dialog.setNegativeButton("취소",new DialogInterface.OnClickListener() {
                    public void onClick(DialogInterface dialog, int whichButton) {
                    }
                });
                dialog.show();

                return false;
            }
        });


        return view;
    }

    public class ViewHolder {

//        private TextView index;
        private TextView mName;
        private TextView mInputDate;
        private TextView mExpiration;
    }

    // 데이터를 추가하는 것을 위해서 만들어 준다.
    public void add(RefItemInfo refItemInfo) {
        mItems.add(refItemInfo);
    }
}

